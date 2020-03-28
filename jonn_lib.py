'''
jonn_lib.py
last updated 03/28/2020
'''

import os
import gc
import dcor
import smtplib
import mysql.connector

import numpy as np
import pandas as pd
import os.path as op
import cvxopt as cvx
import networkx as nx
import seaborn as sns
import scipy.stats as ss
import yahoofinancials as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import matplotlib.gridspec as gridspec

from pypfopt import *
from email import encoders
from arch import arch_model
from functools import partial
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import newton
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from mysql.connector import errorcode
from email.mime.multipart import MIMEMultipart
from email.utils import COMMASPACE, formatdate
from datetime import datetime, timedelta, date

from yahoo_fin import stock_info as si

%matplotlib inline

blue, gold, green, red, purple, brown = sns.color_palette('colorblind', 6)

def daterange(start_date, end_date):
    ''' 
    daterange(date(2005, 12, 31), date(2020, 3, 25))
    creates range between 2 dates given
    '''
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
        
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        gc.collect()
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def df_distance_correlation(df):
    '''
    creates distance correlation matrix
    '''
    #initializes an empty DataFrame
    df_dcor = pd.DataFrame(index=df.columns, columns=df.columns)
    
    #initialzes a counter at zero
    k=0
    
    #iterates over the time series of each stock
    for i in stocks:
        
        #stores the ith time series as a vector
        v_i = df.loc[:, i].values
        
        #iterates over the time series of each stock subect to the counter k
        for j in stocks[k:]:
            
            #stores the jth time series as a vector
            v_j = df.loc[:, j].values
            
            #computes the dcor coefficient between the ith and jth vectors
            dcor_val = dcor.distance_correlation(v_i, v_j)
            
            #appends the dcor value at every ij entry of the empty DataFrame
            df_dcor.at[i,j] = dcor_val
            
            #appends the dcor value at every ji entry of the empty DataFrame
            df_dcor.at[j,i] = dcor_val
        
        #increments counter by 1
        k+=1
    #returns a DataFrame of dcor values for every pair of stocks
    return df_dcor

def build_corr_nx(df):
    '''
    builds correlation network given a distance correlation
    aka run on output from df_distance_correlation
    '''
    # converts the distance correlation dataframe to a numpy matrix with dtype float
    cor_matrix = df.values.astype('float')
    
    # Since dcor ranges between 0 and 1, (0 corresponding to independence and 1
    # corresponding to dependence), 1 - cor_matrix results in values closer to 0
    # indicating a higher degree of dependence where values close to 1 indicate a lower degree of 
    # dependence. This will result in a network with nodes in close proximity reflecting the similarity
    # of their respective time-series and vice versa.
    sim_matrix = 1 - cor_matrix
    
    # transforms the similarity matrix into a graph
    G = nx.from_numpy_matrix(sim_matrix)
    
    # extracts the indices (i.e., the stock names from the dataframe)
    stock_names = df.index.values
    
    # relabels the nodes of the network with the stock names
    G = nx.relabel_nodes(G, lambda x: stock_names[x])
    
    # assigns the edges of the network weights (i.e., the dcor values)
    G.edges(data=True)
    
    # copies G
    ## we need this to delete edges or othwerwise modify G
    H = G.copy()
    
    # iterates over the edges of H (the u-v pairs) and the weights (wt)
    for (u, v, wt) in G.edges.data('weight'):
        # selects edges with dcor values less than or equal to 0.33
        if wt >= 1 - 0.325:
            # removes the edges 
            H.remove_edge(u, v)
            
        # selects self-edges
        if u == v:
            # removes the self-edges
            H.remove_edge(u, v)
    
    # returns the final stock correlation network            
    return H

# function to display the network from the distance correlation matrix
def plt_corr_nx(H, title):
    '''
    plots correlation network
    run on output from build_corr_nx
    '''
    # creates a set of tuples: the edges of G and their corresponding weights
    edges, weights = zip(*nx.get_edge_attributes(H, "weight").items())

    # This draws the network with the Kamada-Kawai path-length cost-function.
    # Nodes are positioned by treating the network as a physical ball-and-spring system. The locations
    # of the nodes are such that the total energy of the system is minimized.
    pos = nx.kamada_kawai_layout(H)

    with sns.axes_style('whitegrid'):
        # figure size and style
        plt.figure(figsize=(12, 9))
        plt.title(title, size=16)

        # computes the degree (number of connections) of each node
        deg = H.degree

        # list of node names
        nodelist = []
        # list of node sizes
        node_sizes = []

        # iterates over deg and appends the node names and degrees
        for n, d in deg:
            nodelist.append(n)
            node_sizes.append(d)

        # draw nodes
        nx.draw_networkx_nodes(
            H,
            pos,
            node_color="#DA70D6",
            nodelist=nodelist,
            node_size=np.power(node_sizes, 2.33),
            alpha=0.8,
            font_weight="bold",
        )

        # node label styles
        nx.draw_networkx_labels(H, pos, font_size=13, font_family="sans-serif", font_weight='bold')

        # color map
        cmap = sns.cubehelix_palette(3, as_cmap=True, reverse=True)

        # draw edges
        nx.draw_networkx_edges(
            H,
            pos,
            edge_list=edges,
            style="solid",
            edge_color=weights,
            edge_cmap=cmap,
            edge_vmin=min(weights),
            edge_vmax=max(weights),
        )

        # builds a colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap, 
            norm=plt.Normalize(vmin=min(weights), 
            vmax=max(weights))
        )
        sm._A = []
        plt.colorbar(sm)

        # displays network without axes
        plt.axis("off")
        
def send_mail(send_from, send_to, subject, message, files=[] ,server="smtp.gmail.com", port=587, username='', password='', use_tls=True):
    """Compose and send email with provided info and attachments.
    send_mail('jromero0413@gmail.com', ['jr4001@columbia.edu','jromero0413@gmail.com'],'HI', 'THIS', files=['TP_python_prev.pdf'], server="smtp.gmail.com",port=587, username='jromero0413@gmail.com', password='XXXXXXXXXXX',use_tls=True)
    Args:
        send_from (str): from name
        send_to (str): to name
        subject (str): message title
        message (str): message body
        files (list[str]): list of file paths to be attached to email
        server (str): mail server host name
        port (int): port number
        username (str): server auth username
        password (str): server auth password
        use_tls (bool): use TLS mode
    """
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message))

    for path in files:
        part = MIMEBase('application', "octet-stream")
        with open(path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename="{}"'.format(op.basename(path)))
        msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    if use_tls:
        smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()
    
def create_database(database, password):
    '''
    creates a database
    '''
    cnx = mysql.connector.connect(user='root', password=password, host='localhost', auth_plugin='mysql_native_password')
    cursor = cnx.cursor()
    try:
        cursor.execute("CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(database))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        cursor.close()
        cnx.close()
        exit(1)
    cursor.close()
    cnx.close()
    
def create_table(database, password, table_name, table):
    '''
    creates table in database
    create_table_jonn_db('jonn_db', 'temp_example', "CREATE TABLE `temp_example` ("
    "  `emp_no` int(11) NOT NULL AUTO_INCREMENT,"
    "  `birth_date` date NOT NULL,"
    "  `first_name` varchar(14) NOT NULL,"
    "  `last_name` varchar(16) NOT NULL,"
    "  `gender` enum('M','F') NOT NULL,"
    "  `hire_date` date NOT NULL,"
    "  PRIMARY KEY (`emp_no`)"
    ") ENGINE=InnoDB" )
    '''
    cnx = mysql.connector.connect(user='root', password=password, host='localhost', database=database, auth_plugin='mysql_native_password')
    cursor = cnx.cursor()
    
    try:
        print("Creating table {}: ".format(table_name), end='')
        cursor.execute(table)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("already exists.")
        else:
            print(err.msg)
    else:
        print("OK")
    
    cursor.close()
    cnx.close()
    
def insert_data_db():
    print ('https://dev.mysql.com/doc/connector-python/en/connector-python-example-cursor-transaction.html')
    
def query_data_db():
    print('https://dev.mysql.com/doc/connector-python/en/connector-python-example-cursor-select.html')
    
def drop_corr(df, keep_cols, thresh=0.99):
    '''
    removes column if overly correlated
    '''
    df_corr = df.corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(np.bool))
    to_remove = [column for column in upper.columns if any(upper[column] > thresh)] 
    to_remove = [x for x in to_remove if x not in keep_cols]
    df_corr = df_corr.drop(columns = to_remove)
    return df_corr, to_remove    

def return_live_price(ticker):
    '''
    returns current price of stock
    '''
    return si.get_live_price(ticker)
    
def get_historical_prices(symbol, start_date, end_date, time_interval):
    '''
    get_historical_prices('TLT', '1989-12-31', datetime.now().strftime("%Y-%m-%d"), 'daily')
    uses yahoo financials to download stock prices monthly, weekly or daily
    '''
    obj = yf.YahooFinancials(symbol)
    data = obj.get_historical_price_data(start_date=start_date, end_date=end_date, time_interval=time_interval)
    df = pd.DataFrame(data[symbol]['prices'])
    df = df.rename(columns={'formatted_date':'Date'})
    df = df.set_index(df['Date'], drop=True)
    
    try:
        divs = pd.DataFrame(data[symbol]['eventsData']['dividends']).T
        divs = divs.rename(columns={'formatted_date':'Date','amount':'dividend'})
        divs = divs.set_index(divs['Date'],drop=True)
        df = df.merge(divs['dividend'],left_index=True,right_index=True,how='outer')
        df['dividend'] = df['dividend'].fillna(0)
    except:
        pass
    
    try:
        df['log_total_return']=np.log(df['adjclose']/df['adjclose'].shift(1))
        df['log_div_return']=np.log(df['adjclose']/(df['adjclose']-df['dividend']))
        df['log_price_return']=np.log(df['close']/df['close'].shift(1))
        df['total_return'] = (df['adjclose']-df['adjclose'].shift(1))/df['adjclose'].shift(1)
        df['div_return'] = (df['dividend'])/df['adjclose'].shift(1)
        df['price_return']= (df['close']-df['close'].shift(1))/df['close'].shift(1)
        df = df.add_suffix('_'+symbol)
    except:
        pass
    return df

def garch(data, p=1, o=0, q=1, update_freq=5, **kwds):
	'''
	garch model
	'''
    model = arch_model(data, 'Garch', p=p, o=o, q=q, **kwds)
    res = model.fit(update_freq=update_freq, disp=False)
    return res

def garch_forecast_sim(data, p=1, o=0, q=1, update_freq=5,horizon=30, n_simulations=1000, **kwds):
	'''
	garch forecast
	'''
    np.random.seed(0)
    garch_model = garch(data,p=p,o=o,q=q,update_freq=update_freq) 
    forecasts = garch_model.forecast(horizon=horizon, method='simulation',simulations=n_simulations)
    sim_ser = pd.Series(forecasts.simulations.values[-1,:,-1])
    sim_ser.name = 'garch'    
    return sim_ser

def calc_garch_var(data, p=1, o=0, q=1, update_freq=5, horizon=30, n_simulations=1000, alpha=0.05, **kwds):
	'''
	calculate garch variance
	'''
    sim_ser = garch_forecast_sim(data, p=p, o=o, q=q, update_freq=update_freq,horizon=horizon, n_simulations=n_simulations//horizon, **kwds)
    var = calc_quantile_var(sim_ser, alpha=alpha)
    return var

def tsplot(y, lags=None, figsize=(12, 12), style='bmh'):
	'''
	creates ACF PACF and QQ plots for time series analysis
	'''
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
        
        #y.plot(ax=ts_ax)
        y.plot(ax=ts_ax, marker='o', c='gray', ms=5.,
               markerfacecolor=blue, markeredgecolor='white', markeredgewidth=.25)        
        ts_ax.set_title(f'{y.name}')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax, marker='X', markerfacecolor=blue,
                  markeredgecolor='k', markeredgewidth=0.25)
        qq_ax.set_title('QQ Plot')        
        plt.tight_layout()
    return 

def BlackScholes(S,K,r,T,sigma, voltype='Normal', option = 'call'):
    """Return price of swaption in BlackScholes model
    Inputs:
    S = spot (current value of forward swap rate)
    K = strike
    sigma = volatility (in normal or lognormal units)
    T = expiry
    r = interest rates
    voltype = one of 'Normal' or 'Lognormal' """
    if voltype=='Normal':
        moneyness = S-K
        atMaturityStdev = sigma*np.sqrt(T)
        scaledMoneyness = moneyness/atMaturityStdev
        return (moneyness * norm.cdf(scaledMoneyness)+atMaturityStdev * norm.pdf(scaledMoneyness))
    elif voltype=='Lognormal':
        d1 = (np.log(1.0*S/K)+(r+sigma**2/2.0)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if  option == 'call':
            return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) 
        else:
            return (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise Exception("Unsupported option volatility type, should be 'Normal' or 'Lognormal', '" + voltype + "' entered."  )
        
def impliedVol(Opt_price, S0,K,r,T, vol_guess , voltype = 'Lognormal'):
	'''
	calculates implied volatility given
	'''
    return newton(func = lambda sigma:BlackScholes(S0,K,r,T, sigma, voltype='Lognormal')-Opt_price,x0 = vol_guess,tol = 1e-8)
        
def BSMC(S0, K, r, T, sig , N , payoff):
	'''
	BS Monte Carlo
	'''
    np.random.seed(seed=3) 
    W = ss.norm.rvs( loc=(r - 0.5 * sig**2)*T, scale=np.sqrt(T)*sig, size=N )
    S_T = S0 * np.exp(W)
    if payoff =="call":
        V = np.average(np.maximum(S_T-K, 0.0)*np.exp(-r*T))
    else:
        V = np.average(np.maximum(K-S_T, 0.0)*np.exp(-r*T))
    return V

def BSMCAnti(S0, K, r, T, sig , N , payoff):
	'''
	calculates BS using antithetic 
	'''
    np.random.seed(seed=3) 
    W = ss.norm.rvs( loc=0, scale=np.sqrt(T)*sig, size=int(N/2) )
    volBias = np.sqrt(T)*sig/np.std(W)
    S_T = np.concatenate([S0 * np.exp((r - 0.5 * sig**2)*T+W*volBias),S0 * np.exp((r - 0.5 * sig**2)*T-W*volBias)])
    
    if payoff =="call":
        V = np.average(np.maximum(S_T-K, 0.0)*np.exp(-r*T))
    else:
        V = np.average(np.maximum(K-S_T, 0.0)*np.exp(-r*T))
    return V

def tree_call(S0, K, r, T, sig , N , payoff):
	'''
	calucalcutes binomial pricing
	'''

    dT = float(T) / N                             # Delta t
    u = np.exp(sig * np.sqrt(dT))                 # up factor
    d = 1.0 / u                                   # down factor 
    N=int(N)
    V = np.zeros(N+1)                             # initialize the price vector
    S_T = np.array( [(S0 * u**j * d**(N - j)) for j in range(N + 1)] )  # price S_T at time T

    a = np.exp(r * dT)    # risk free compounded return
    p = (a - d)/ (u - d)  # risk neutral up probability
    q = 1.0 - p           # risk neutral down probability   

    if payoff =="call":
        V[:] = np.maximum(S_T-K, 0.0)
    else:
        V[:] = np.maximum(K-S_T, 0.0)

    for i in range(N-1, -1, -1):
        V[:-1] = np.exp(-r*dT) * (p * V[1:] + q * V[:-1])    # the price vector is overwritten at each step
    return V[0]

def markowitz_opt(ret_vec, covar_mat, max_risk):
    '''
    markowitz_opt([0.07, 0.06, 0.0], [[0.2,0,0],[0,0.3,0],[0,0,0]], 0.07)
    Finds the best return for a given level of risks usesing cvxopt
    '''
    U,V = np.linalg.eig(covar_mat)
    U[U<0] = 0
    Usqrt = np.sqrt(U)
    A = np.dot(np.diag(Usqrt), V.T)

    G1temp = np.zeros((A.shape[0]+1, A.shape[1]))
    G1temp[1:, :] = -A
    h1temp = np.zeros((A.shape[0]+1, 1))
    h1temp[0] = max_risk

    ret_c = len(ret_vec)
    for i in np.arange(ret_c):
        ei = np.zeros((1, ret_c))
        ei[0, i] = 1
        if i == 0:
            G2temp = [cvx.matrix(-ei)]
            h2temp = [cvx.matrix(np.zeros((1,1)))]
        else:
            G2temp += [cvx.matrix(-ei)]
            h2temp += [cvx.matrix(np.zeros((1,1)))]

    Ftemp = np.ones((1, ret_c))
    F = cvx.matrix(Ftemp)
    g = cvx.matrix(np.ones((1,1)))

    G = [cvx.matrix(G1temp)] + G2temp
    H = [cvx.matrix(h1temp)] + h2temp

    cvx.solvers.options['show_progress'] = False
    sol = cvx.solvers.socp(
        -cvx.matrix(ret_vec), 
        Gq=G, hq=H, A=F, b=g)
    xsol = np.array(sol['x'])
    return xsol