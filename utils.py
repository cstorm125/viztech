import numpy as np
import pandas as pd
import scipy.stats as st
from plotnine import *
import seaborn as sns
from sklearn.linear_model import LinearRegression

#######################################################################
'''
Snippet for plotnine with thai font by @korakot
https://gist.github.com/korakot/01d181229b21411b0a20784e0ca20d3d
'''
import matplotlib
# !wget https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf -q
# !cp thsarabunnew-webfont.ttf /usr/share/fonts/truetype/
matplotlib.font_manager._rebuild()
matplotlib.rc('font', family='TH Sarabun New')
theme_set(theme_minimal(11, 'TH Sarabun New'))
#######################################################################

#cleaning
def check_missing(df):
    per_missing = df.isnull().mean()
    missing_df = pd.DataFrame({'col_name': df.columns, 'per_missing': per_missing})
    missing_df = missing_df.sort_values('per_missing',ascending=False).reset_index(drop=True)
    missing_df['rnk'] = missing_df.index.map(lambda x: str(x).zfill(2)+'_') + missing_df.col_name
    return missing_df

def check_mode(df):
    mode_df = []
    for col in df.columns:
        x = df[col].value_counts()
        mode_df.append({'col':col, 'value':x.index[0], 'per_mode': list(x)[0]/df.shape[0],
                       'nb_value':len(x)})
    mode_df = pd.DataFrame(mode_df)[['col','value','per_mode','nb_value']]\
        .sort_values('per_mode',ascending=False)
    return mode_df.reset_index(drop=True)

def remove_outliers(df,col):
    q1 = np.percentile(df[col], 25)
    q3 = np.percentile(df[col], 75)
    iqr = q3-q1
    df = df[(df[col] < q3+1.5*iqr)&(df[col] > q1-1.5*iqr)]
    return df.reset_index(drop=True)

def value_dist(df,col):
    x = pd.DataFrame(df[col].value_counts()).reset_index()
    x.columns = ['value','cnt']
    x['per'] = x.cnt / x.cnt.sum()
    return x

def otherify(df,col, th=0.03, retain=['NA']):
    value_df = value_dist(df,col)
    other_cols = list(value_df[value_df.per<th].value)
    df[col] = df[col].map(lambda x: 'others' if (x in other_cols and x not in retain) else x)
    return df

def replace_dict(x, d):
    for key,value_list in d.items():
        for v in value_list:
            if v in str(x).lower():
                return key
    return 'others'

#visualization
def thai_text(font_size=11): 
    #thanks to @korakot for Thai font with plotnine
    return theme(text=element_text(size=font_size,  family='TH Sarabun New'))

def cat_plot(df,col):
    g = (ggplot(df,aes(x=col)) + 
         geom_bar(stat='bin', #histogram
                  binwidth=0.5, #histogram binwidth
                  bins=len(df[col].unique())) + #how many bins
         coord_flip()
#          theme(axis_text_x=element_blank())
        )
    return g

def numcat_plot(df,num,cat, no_outliers=True, geom=geom_boxplot()):
    if no_outliers:
        new_df = remove_outliers(df,num)
    else:
        new_df = df.copy()
    g = (ggplot(new_df, aes(x=cat,y=num)) +
         geom 
        )
    return g

def numdist_plot(df, num,cat, geom=geom_density(alpha=0.5), no_outliers=True):
    if no_outliers:
        new_df = remove_outliers(df,num)
    else:
        new_df = df.copy()
    g = (ggplot(new_df,aes(x=num, fill=cat)) +
          geom 
        )
    return g

def catcat_plot(df, cat_dep, cat_ind):
    plot_df = df.copy()
    plot_df['cnt'] = 1
    df_total = plot_df[[cat_ind,'cnt']].groupby(cat_ind).count().reset_index()
    df_agg = plot_df[[cat_dep,cat_ind,'cnt']].groupby([cat_dep,cat_ind]).count().reset_index()
    df_agg = df_agg.merge(df_total,on='color')
    df_agg['per'] = df_agg.cnt_x / df_agg.cnt_y
    g = (ggplot(df_agg,aes(x=cat_dep, y='per',fill=cat_dep)) + 
         geom_col() + 
         geom_text(aes(x=cat_dep,y='per',label='cnt_x')) +
         theme(axis_text_x = element_blank()) +
         facet_wrap(f'~{cat_ind}')) + theme(panel_spacing_x=0.5)
    return g

def value_dist_plot(df,bins=30):
    num_m = df.melt()
    g = (ggplot(num_m,aes(x='value')) +
         geom_bar(stat='bin', bins=bins) +
         facet_wrap('~variable', scales='free') + #facetting by variable
         theme_minimal() + theme(panel_spacing_x=0.5)
        )
    return g

def jointplot(df,col_x, col_y, no_outliers=True, kind='reg'): #'scatter','resid','reg','hex','kde','point'
    if no_outliers:
        new_df = remove_outliers(df,col_x)
        new_df = remove_outliers(new_df,col_y)
    else:
        new_df = df.copy()
    return sns.jointplot(new_df[col_x],new_df[col_y],kind=kind)
    
def qq_plot(df,col):
    qq, reg = calc_qq(df,col)
    g = (ggplot(qq,aes(x='theoretical_q',y='sample_q')) + 
        geom_point() + #plot points
        geom_abline(slope=1,intercept=0,color='red') + #perfectly normal line
        stat_function(fun=lambda x: x*reg.coef_[0][0]) + #linear estimation
        ggtitle(f'y= {np.round(reg.coef_[0][0],2)} * x')+ #display equation
        labs(x='Theoretical Quantiles (normalized)', y='Sample Qunatiles (normalized)'))
    return g

def boxcox_plot(df, col, ls = [i/10 for i in range(-30,31,5)]):
    lamb_df = boxcox_lamb_df(df[col],ls)
    g = (ggplot(lamb_df, aes(x='lamb',y='coef',group=1)) + 
         geom_point() + geom_line())
    return g

#transformation
def calc_qq(df,col):
    sample_qs = [(np.percentile(df[col],i)-np.mean(df[col]))/np.std(df[col]) for i in range(5,100,5)]
    theoretical_qs = [st.norm.ppf(i/100) for i in range(5,100,5)]
    qq = pd.DataFrame({'sample_q':sample_qs,'theoretical_q':theoretical_qs})
    reg = LinearRegression(fit_intercept=False).fit(np.array(qq['theoretical_q'])[:,None], 
                                 np.array(qq['sample_q'])[:,None])
    return qq, reg

def boxcox(ser,lamb=0):
    ser+= 1 - ser.min()
    if lamb==0: 
        return np.log(ser)
    else:
        return (ser**lamb - 1)/lamb
    
def boxcox_lamb_df(ser, ls = [i/10 for i in range(-30,31,5)]):
    coefs = []
    for l in ls:
        df = pd.DataFrame.from_dict({'val': boxcox(ser,l)})
        qq, reg = calc_qq(df,'val')
        coefs.append(reg.coef_.squeeze().item())
    return pd.DataFrame({'lamb':ls,'coef':coefs})

def boxcox_lamb(ser, ls = [i/10 for i in range(-30,31,5)]):
    df = boxcox_lamb_df(ser,ls)
    return df.lamb[df.coef.idxmax()]
