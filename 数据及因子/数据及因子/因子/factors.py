import pandas as pd
import numpy as np
import bottleneck as bn

def fm2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        lowmin=low.rolling(ma_len).min()
        cpl=cl/lowmin-1
        cpl_sq_ma=(cpl**2).rolling(ma_len).mean()
        factor=cpl_sq_ma**0.5
        stock_data['fm2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm2_'+str(int(ma_len))]=stock_data['fm2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm2_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        highmax=high.rolling(ma_len).max()
        cph=cl/highmax-1
        cph_sq_ma=(cph**2).rolling(ma_len).mean()
        factor=cph_sq_ma**0.5
        stock_data['fm2_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm2_2_'+str(int(ma_len))]=stock_data['fm2_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm2_2_'+str(int(ma_len)))
    return stock_data,factor_list


def fm2_3(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        lowmin=low.rolling(ma_len).min()
        highmax=high.rolling(ma_len).max()
        mid=(highmax+lowmin)/2
        cpm=cl/mid-1
        cpm_sq_ma=(cpm**2).rolling(ma_len).mean()
        factor=cpm_sq_ma**0.5
        stock_data['fm2_3_'+str(int(ma_len))]=factor.copy()
        stock_data['fm2_3_'+str(int(ma_len))]=stock_data['fm2_3_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm2_3_'+str(int(ma_len)))
    return stock_data,factor_list

def fm2_4(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    mid=(high+low)/2
    for ma_len in [2**i for i in period_id_list]:
        cpl=vwap/mid-1
        cpl_sq_ma=(cpl**2).rolling(ma_len).mean()
        factor=cpl_sq_ma**0.5
        stock_data['fm2_4_'+str(int(ma_len))]=factor.copy()
        stock_data['fm2_4_'+str(int(ma_len))]=stock_data['fm2_4_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm2_4_'+str(int(ma_len)))
    return stock_data,factor_list

def fm11(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        amt_ma=amount.rolling(ma_len).mean()
        amt_rank=bn.move_rank(amt_ma,window=ma_len,min_count=1)
        stock_data['fm11_'+str(int(ma_len))]=amt_rank.copy()
        stock_data['fm11_'+str(int(ma_len))]=stock_data['fm11_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm11_'+str(int(ma_len)))
    return stock_data,factor_list

def fm12(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        vol_short=vol.rolling(ma_len).mean()
        vol_long=vol.rolling(ma_len*2).mean()
        vol_ratio=vol_short/vol_long
        stock_data['fm12_'+str(int(ma_len))]=vol_ratio.copy()
        stock_data['fm12_'+str(int(ma_len))]=stock_data['fm12_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm12_'+str(int(ma_len)))
    return stock_data,factor_list

def fm12_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        vol_short=vol
        vol_long=vol.rolling(ma_len).mean()
        vol_ratio=vol_short/vol_long
        stock_data['fm12_2_'+str(int(ma_len))]=vol_ratio.copy()
        stock_data['fm12_2_'+str(int(ma_len))]=stock_data['fm12_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm12_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm14(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        if ma_len==1:
            short_len=1
        else:
            short_len=int(ma_len/2)
        vol_short=vol.rolling(short_len).mean()
        vol_mean=vol_short.rolling(ma_len).mean()
        vol_std=vol_short.rolling(ma_len).std()
        vol_ratio=(vol_short-vol_mean)/vol_std
        stock_data['fm14_'+str(int(ma_len))]=vol_ratio.copy()
        stock_data['fm14_'+str(int(ma_len))]=stock_data['fm14_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm14_'+str(int(ma_len)))
    return stock_data,factor_list

def fm18(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        clsma=cl.rolling(ma_len).mean()
        hl=high-low
        factor=hl/clsma
        stock_data['fm18_'+str(int(ma_len))]=factor.copy()
        stock_data['fm18_'+str(int(ma_len))]=stock_data['fm18_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm18_'+str(int(ma_len)))
    return stock_data,factor_list

def fm21(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        cl_mean=cl.rolling(ma_len).mean()
        cl_std=cl.rolling(ma_len).std()
        norm=(cl-cl_mean)/cl_std
        norm.loc[np.abs(cl_std)<1e-5]=0
        factor=bn.move_argmax(norm,window=ma_len,min_count=1)
        stock_data['fm21_'+str(int(ma_len))]=factor.copy()
        stock_data['fm21_'+str(int(ma_len))]=stock_data['fm21_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm21_'+str(int(ma_len)))
    return stock_data,factor_list

def fm21_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        cl_mean=cl.rolling(ma_len).mean()
        cl_std=cl.rolling(ma_len).std()
        norm=(cl-cl_mean)/cl_std
        norm.loc[np.abs(cl_std)<1e-5]=0
        factor=bn.move_argmin(norm,window=ma_len,min_count=1)
        stock_data['fm21_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm21_2_'+str(int(ma_len))]=stock_data['fm21_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm21_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm21_3(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        cl_mean=cl.rolling(ma_len).mean()
        cl_std=cl.rolling(ma_len).std()
        norm=(cl-cl_mean)/cl_std
        norm.loc[np.abs(cl_std)<1e-5]=0
        factor=bn.move_argmax(norm,window=ma_len,min_count=1)-bn.move_argmin(norm,window=ma_len,min_count=1)
        stock_data['fm21_3_'+str(int(ma_len))]=factor.copy()
        stock_data['fm21_3_'+str(int(ma_len))]=stock_data['fm21_3_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm21_3_'+str(int(ma_len)))
    return stock_data,factor_list

def fm25(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    vol_diff=vol.diff()
    sign_vr=np.sign(vol_diff)*np.sign(ret)
    for ma_len in [2**i for i in period_id_list]:
        factor=pd.Series(sign_vr).rolling(ma_len).mean()
        stock_data['fm25_'+str(int(ma_len))]=factor.copy()
        stock_data['fm25_'+str(int(ma_len))]=stock_data['fm25_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm25_'+str(int(ma_len)))
    return stock_data,factor_list

def fm25_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    vol_diff=vol.diff()
    sign_vr=vol*0
    sign_vr.loc[(vol_diff>0)&(ret>0)]=1
    sign_vr.loc[(vol_diff>0)&(ret<0)]=-1
    for ma_len in [2**i for i in period_id_list]:
        factor=pd.Series(sign_vr).rolling(ma_len).mean()
        stock_data['fm25_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm25_2_'+str(int(ma_len))]=stock_data['fm25_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm25_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm25_3(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    vol_diff=vol.diff()
    sign_vr=vol*0
    sign_vr.loc[(vol_diff>0)&(ret>0)]=ret.loc[(vol_diff>0)&(ret>0)]
    sign_vr.loc[(vol_diff>0)&(ret<0)]=ret.loc[(vol_diff>0)&(ret<0)]
    for ma_len in [2**i for i in period_id_list]:
        factor=pd.Series(sign_vr).rolling(ma_len).mean()
        stock_data['fm25_3_'+str(int(ma_len))]=factor.copy()
        stock_data['fm25_3_'+str(int(ma_len))]=stock_data['fm25_3_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm25_3_'+str(int(ma_len)))
    return stock_data,factor_list

def fm25_4(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    vol_diff=vol.diff()
    sign_vr=vol*0
    sign_vr.loc[(vol_diff<0)&(ret>0)]=ret.loc[(vol_diff<0)&(ret>0)]
    sign_vr.loc[(vol_diff<0)&(ret<0)]=ret.loc[(vol_diff<0)&(ret<0)]
    for ma_len in [2**i for i in period_id_list]:
        factor=pd.Series(sign_vr).rolling(ma_len).mean()
        stock_data['fm25_4_'+str(int(ma_len))]=factor.copy()
        stock_data['fm25_4_'+str(int(ma_len))]=stock_data['fm25_4_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm25_4_'+str(int(ma_len)))
    return stock_data,factor_list

def fm25_5(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    vol_diff=vol.diff()
    sign_vr=vol*0
    sign_vr.loc[(vol_diff<0)&(ret>0)]=-ret.loc[(vol_diff<0)&(ret>0)]
    sign_vr.loc[(vol_diff>0)&(ret>0)]=ret.loc[(vol_diff>0)&(ret>0)]
    for ma_len in [2**i for i in period_id_list]:
        factor=pd.Series(sign_vr).rolling(ma_len).mean()
        stock_data['fm25_5_'+str(int(ma_len))]=factor.copy()
        stock_data['fm25_5_'+str(int(ma_len))]=stock_data['fm25_5_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm25_5_'+str(int(ma_len)))
    return stock_data,factor_list

def fm25_6(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    vol_diff=vol.diff()
    sign_vr=vol*0
    sign_vr.loc[(vol_diff<0)&(ret<0)]=-ret.loc[(vol_diff<0)&(ret<0)]
    sign_vr.loc[(vol_diff>0)&(ret<0)]=ret.loc[(vol_diff>0)&(ret<0)]
    for ma_len in [2**i for i in period_id_list]:
        factor=pd.Series(sign_vr).rolling(ma_len).mean()
        stock_data['fm25_6_'+str(int(ma_len))]=factor.copy()
        stock_data['fm25_6_'+str(int(ma_len))]=stock_data['fm25_6_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm25_6_'+str(int(ma_len)))
    return stock_data,factor_list

def fm25_7(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    vol_diff=vol.diff()
    sign_vr=vol*0
    sign_vr.loc[(vol_diff<0)&(ret>0)]=-ret.loc[(vol_diff<0)&(ret>0)]
    sign_vr.loc[(vol_diff>0)&(ret>0)]=ret.loc[(vol_diff>0)&(ret>0)]
    sign_vr.loc[(vol_diff<0)&(ret<0)]=-ret.loc[(vol_diff<0)&(ret<0)]
    sign_vr.loc[(vol_diff>0)&(ret<0)]=ret.loc[(vol_diff>0)&(ret<0)]
    for ma_len in [2**i for i in period_id_list]:
        factor=pd.Series(sign_vr).rolling(ma_len).mean()
        stock_data['fm25_7_'+str(int(ma_len))]=factor.copy()
        stock_data['fm25_7_'+str(int(ma_len))]=stock_data['fm25_7_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm25_7_'+str(int(ma_len)))
    return stock_data,factor_list

def fm32(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    hdis=high-vwap
    ldis=vwap-low
    hlrate=hdis/ldis
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_rank(hlrate,window=ma_len,min_count=1)
        stock_data['fm32_'+str(int(ma_len))]=factor.copy()
        stock_data['fm32_'+str(int(ma_len))]=stock_data['fm32_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm32_'+str(int(ma_len)))
    return stock_data,factor_list

def fm33(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    hlrate=high/low
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_rank(hlrate,window=ma_len,min_count=1)
        stock_data['fm33_'+str(int(ma_len))]=factor.copy()
        stock_data['fm33_'+str(int(ma_len))]=stock_data['fm33_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm33_'+str(int(ma_len)))
    return stock_data,factor_list

def fm33_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    hlrate=high/low
    for ma_len in [2**i for i in period_id_list]:
        if ma_len==1:
            short_len=1
        else:
            short_len=int(ma_len/2)
        hl_ma=hlrate.rolling(int(ma_len/2)).mean()
        factor=bn.move_rank(hl_ma,window=ma_len,min_count=1)
        stock_data['fm33_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm33_2_'+str(int(ma_len))]=stock_data['fm33_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm33_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm34(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        highmax=bn.move_max(high,window=ma_len,min_count=1)
        lowmin=bn.move_min(low,window=ma_len,min_count=1)
        mrm=highmax/lowmin
        factor=bn.move_rank(mrm,window=ma_len,min_count=1)
        stock_data['fm34_'+str(int(ma_len))]=factor.copy()
        stock_data['fm34_'+str(int(ma_len))]=stock_data['fm34_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm34_'+str(int(ma_len)))
    return stock_data,factor_list

def fm49(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        highmax=bn.move_max(high,window=ma_len,min_count=1)
        cph=cl/highmax-1
        cphsq=cph**2 
        cphsq_ma=cphsq.rolling(ma_len).mean()
        factor=cphsq_ma**0.5
        stock_data['fm49_'+str(int(ma_len))]=factor.copy()
        stock_data['fm49_'+str(int(ma_len))]=stock_data['fm49_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm49_'+str(int(ma_len)))
    return stock_data,factor_list

def fm61(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    amt_ratio=amount/amount.rolling(48).mean()
    for ma_len in [2**i for i in period_id_list]:
        if ma_len==1:
            short_len=1
        else:
            short_len=int(ma_len/2)
        ret_diff=ret.rolling(short_len).mean()-ret.rolling(ma_len).mean()
        amount_std=amt_ratio.rolling(ma_len).std()
        factor=ret_diff/amount_std  
        factor.loc[np.abs(amount_std)<1e-5]=0
        stock_data['fm61_'+str(int(ma_len))]=factor.copy()
        stock_data['fm61_'+str(int(ma_len))]=stock_data['fm61_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm61_'+str(int(ma_len)))
    return stock_data,factor_list

def fm62(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    amt_ratio=amount/amount.rolling(48).mean()
    for ma_len in [2**i for i in period_id_list]:
        if ma_len==1:
            short_len=1
        else:
            short_len=int(ma_len/2)
        ret_diff=ret.rolling(short_len).mean()-ret.rolling(ma_len).mean()
        amount_mean=amt_ratio.rolling(ma_len).mean()
        factor=ret_diff/amount_mean
        factor.loc[np.abs(amount_mean)<1e-5]=0
        stock_data['fm62_'+str(int(ma_len))]=factor.copy()
        stock_data['fm62_'+str(int(ma_len))]=stock_data['fm62_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm62_'+str(int(ma_len)))
    return stock_data,factor_list

def fm65(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    hl=high-low
    hlv=hl*vol
    for ma_len in [2**i for i in period_id_list]:
        hlma=hl.rolling(ma_len).mean()
        vma=vol.rolling(ma_len).mean()
        hlvma1=hlma*vma 
        hlvma2=hlv.rolling(ma_len).mean()
        factor=(hlvma1-hlvma2)/(hlvma1+hlvma2)
        factor.loc[np.abs(hlvma1+hlvma2)<1e-5]=0
        stock_data['fm65_'+str(int(ma_len))]=factor.copy()
        stock_data['fm65_'+str(int(ma_len))]=stock_data['fm65_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm65_'+str(int(ma_len)))
    return stock_data,factor_list

def fm66(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    hl=high+low
    hlv=hl*vol
    for ma_len in [2**i for i in period_id_list]:
        hlma=hl.rolling(ma_len).mean()
        vma=vol.rolling(ma_len).mean()
        hlvma1=hlma*vma 
        hlvma2=hlv.rolling(ma_len).mean()
        factor=(hlvma1-hlvma2)/(hlvma1+hlvma2)
        factor.loc[np.abs(hlvma1+hlvma2)<1e-5]=0
        stock_data['fm66_'+str(int(ma_len))]=factor.copy()
        stock_data['fm66_'+str(int(ma_len))]=stock_data['fm66_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm66_'+str(int(ma_len)))
    return stock_data,factor_list

def fm69(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    hl_add=high+low
    hl_diff=high-low
    vol_ratio=vol/vol.rolling(48).mean()
    for ma_len in [2**i for i in period_id_list]:
        hlma=hl_diff.rolling(ma_len).mean()
        hladdma=hl_add.rolling(ma_len).mean()
        vma=vol_ratio.rolling(ma_len).mean()
        mid=(hl_add-hladdma)/hladdma
        factor=mid*vma*hl_diff/hlma
        factor.loc[np.abs(hlma)<1e-5]=0
        stock_data['fm69_'+str(int(ma_len))]=factor.copy()
        stock_data['fm69_'+str(int(ma_len))]=stock_data['fm69_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm69_'+str(int(ma_len)))
    return stock_data,factor_list

def fm70(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    hl_add=high+low
    hl_diff=high-low
    for ma_len in [2**i for i in period_id_list]:
        hlma=hl_diff.rolling(ma_len).mean()
        hladdma=hl_add.rolling(ma_len).mean()
        mid=(hl_diff-hlma)/hlma 
        mid.loc[np.abs(hlma)<1e-5]=0
        factor=mid*hl_add/hladdma
        stock_data['fm70_'+str(int(ma_len))]=factor.copy()
        stock_data['fm70_'+str(int(ma_len))]=stock_data['fm70_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm70_'+str(int(ma_len)))
    return stock_data,factor_list

def fm71(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        hmax=high.rolling(ma_len).max()
        vma=vol.rolling(ma_len).mean()
        amtma=amount.rolling(ma_len).mean()
        factor=amtma/vma/hmax
        stock_data['fm71_'+str(int(ma_len))]=factor.copy()
        stock_data['fm71_'+str(int(ma_len))]=stock_data['fm71_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm71_'+str(int(ma_len)))
    return stock_data,factor_list

def fm77(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        clsma=cl.rolling(ma_len).mean()
        volma=vol.rolling(ma_len).mean()
        cvma=(cl*vol).rolling(ma_len).mean()
        factor=cvma/(clsma*volma)
        stock_data['fm77_'+str(int(ma_len))]=factor.copy()
        stock_data['fm77_'+str(int(ma_len))]=stock_data['fm77_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm77_'+str(int(ma_len)))
    return stock_data,factor_list


def fm78(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        volrank=bn.move_rank(vol,window=ma_len,min_count=1)
        stock_data['fm78_'+str(int(ma_len))]=volrank.copy()
        stock_data['fm78_'+str(int(ma_len))]=stock_data['fm78_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm78_'+str(int(ma_len)))
    return stock_data,factor_list

def fm79(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    trend=(cl/low+high/op)/2.0
    for ma_len in [2**i for i in period_id_list]:
        trendmax=trend.rolling(ma_len).max()
        factor=trend/trendmax
        stock_data['fm79_'+str(int(ma_len))]=factor.copy()
        stock_data['fm79_'+str(int(ma_len))]=stock_data['fm79_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm79_'+str(int(ma_len)))
    return stock_data,factor_list

def fm80(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    trend=(cl/low+high/op)/2
    for ma_len in [2**i for i in period_id_list]:
        trend_max=trend.rolling(ma_len).max()
        trend_min=trend.rolling(ma_len).min()
        factor=(trend_max-trend_min)/(trend_max+trend_min)
        stock_data['fm80_'+str(int(ma_len))]=factor.copy()
        stock_data['fm80_'+str(int(ma_len))]=stock_data['fm80_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm80_'+str(int(ma_len)))
    return stock_data,factor_list

def fm81(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data.open
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    factor_list=[]
    trend=(cl/low+high/op)/2.0
    for ma_len in [2**i for i in period_id_list]:
        trendmin=trend.rolling(ma_len).min()
        factor=trend/trendmin
        stock_data['fm81_'+str(int(ma_len))]=factor.copy()
        stock_data['fm81_'+str(int(ma_len))]=stock_data['fm81_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm81_'+str(int(ma_len)))
    return stock_data,factor_list

def fm82(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        vwapr1=(vwap/vwap.shift(ma_len))**(1/ma_len)
        vwapr2=(vwap/vwap.shift(1)).rolling(ma_len).mean()
        factor=(vwapr1-vwapr2)*10000
        stock_data['fm82_'+str(int(ma_len))]=factor.copy()
        stock_data['fm82_'+str(int(ma_len))]=stock_data['fm82_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm82_'+str(int(ma_len)))
    return stock_data,factor_list

def fm83(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    for ma_len in [2**i for i in period_id_list]:
        hmax=high.rolling(ma_len).max()
        lmin=low.rolling(ma_len).min()
        cmax=cl.rolling(ma_len).max()
        cmin=cl.rolling(ma_len).min()
        pos1=(cl-lmin)/(hmax-lmin)
        pos2=(cl-cmin)/(cmax-cmin)
        pos1.loc[np.abs(hmax-lmin)<1e-5]=0
        pos2.loc[np.abs(cmax-cmin)<1e-5]=0
        factor=pos1**pos2-pos2**pos1
        stock_data['fm83_'+str(int(ma_len))]=factor.copy()
        stock_data['fm83_'+str(int(ma_len))]=stock_data['fm83_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm83_'+str(int(ma_len)))
    return stock_data,factor_list

def fm87(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    for ma_len in [2**i for i in period_id_list]:
        hmax=high.rolling(ma_len).max()
        lmin=low.rolling(ma_len).min()
        cmax=cl.rolling(ma_len).max()
        cmin=cl.rolling(ma_len).min()
        pos1=(cl-lmin)/(hmax-lmin)
        pos2=(cl-cmin)/(cmax-cmin)
        pos1.loc[np.abs(hmax-lmin)<1e-5]=0
        pos2.loc[np.abs(cmax-cmin)<1e-5]=0
        factor=(pos1-pos2)/(pos1+pos2)
        factor.loc[np.abs(pos1+pos2)<1e-5]=0
        stock_data['fm87_'+str(int(ma_len))]=factor.copy()
        stock_data['fm87_'+str(int(ma_len))]=stock_data['fm87_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm87_'+str(int(ma_len)))
    return stock_data,factor_list

def fm92(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=np.array(stock_data.vol)
    factor=np.zeros(len(stock_data))
    for ma_len in [2**i for i in period_id_list]:
        factor[:]=0
        vwaprk=bn.move_rank(np.array(vwap),window=ma_len,min_count=1)
        for ℹ in range(ma_len,len(stock_data)):
            vol_wind=vol[(i-ma_len+1):(i+1)]
            vol_argmax=np.argmax(vol_wind)
            factor[i]=vwaprk[(i-ma_len+1):(i+1)][vol_argmax]
        stock_data['fm92_'+str(int(ma_len))]=factor.copy()
        stock_data['fm92_'+str(int(ma_len))]=stock_data['fm92_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm92_'+str(int(ma_len)))
    return stock_data,factor_list

def fm92_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    factor=np.zeros(len(stock_data))
    for ma_len in [2**i for i in period_id_list]:
        if ma_len/2<1:
            vol_len=1
        else:
            vol_len=int(ma_len/2)
        vol_array=np.array(vol.rolling(vol_len).mean())
        factor[:]=0
        vwaprk=bn.move_rank(np.array(vwap),window=ma_len,min_count=1)
        for ℹ in range(ma_len,len(stock_data)):
            vol_wind=vol_array[(i-ma_len+1):(i+1)]
            vol_argmax=np.argmax(vol_wind)
            factor[i]=vwaprk[(i-ma_len+1):(i+1)][vol_argmax]
        stock_data['fm92_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm92_2_'+str(int(ma_len))]=stock_data['fm92_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm92_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm93(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    for ma_len in [2**i for i in period_id_list]:
        rawret=cl/cl.shift(ma_len)
        vma=vol.rolling(ma_len).mean()
        vrate=vol/vma 
        vrate.loc[np.abs(vma)<1e-5]=0
        factor=vrate.rolling(ma_len).std()*rawret
        stock_data['fm93_'+str(int(ma_len))]=factor.copy()
        stock_data['fm93_'+str(int(ma_len))]=stock_data['fm93_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm93_'+str(int(ma_len)))
    return stock_data,factor_list

def fm94(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=np.array(stock_data.vol)
    factor=np.zeros(len(stock_data))
    for ma_len in [2**i for i in period_id_list]:
        factor[:]=0
        volrk=bn.move_rank(np.array(vol),window=ma_len,min_count=1)
        for ℹ in range(ma_len,len(stock_data)):
            vwap_wind=vwap[(i-ma_len+1):(i+1)]
            vwap_argmax=np.argmax(vwap_wind)
            factor[i]=volrk[(i-ma_len+1):(i+1)][vwap_argmax]
        stock_data['fm94_'+str(int(ma_len))]=factor.copy()
        stock_data['fm94_'+str(int(ma_len))]=stock_data['fm94_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm94_'+str(int(ma_len)))
    return stock_data,factor_list


def fm102(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=np.array(stock_data.vol)
    amount=stock_data.amount
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_argmax(amount.values,window=ma_len,min_count=1)/ma_len
        stock_data['fm102_'+str(int(ma_len))]=factor.copy()
        stock_data['fm102_'+str(int(ma_len))]=stock_data['fm102_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm102_'+str(int(ma_len)))
    return stock_data,factor_list

def fm104(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=np.array(stock_data.vol)
    amount=stock_data.amount
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_argmax(cl.values,window=ma_len,min_count=1)/ma_len-bn.move_argmax(amount.values,window=ma_len,min_count=1)/ma_len
        stock_data['fm104_'+str(int(ma_len))]=factor.copy()
        stock_data['fm104_'+str(int(ma_len))]=stock_data['fm104_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm104_'+str(int(ma_len)))
    return stock_data,factor_list

def fm104_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=np.array(stock_data.vol)
    amount=stock_data.amount
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_argmin(cl.values,window=ma_len,min_count=1)/ma_len-bn.move_argmax(amount.values,window=ma_len,min_count=1)/ma_len
        stock_data['fm104_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm104_2_'+str(int(ma_len))]=stock_data['fm104_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm104_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm106(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    a=np.zeros(len(vol))
    b=np.zeros(len(vol))
    vol_diff=vol.diff()
    for ma_len in [2**i for i in period_id_list]:
        a[:]=0
        b[:]=0
        a[vol_diff>0]=1
        b[vol_diff<0]=1
        abv=a-b
        factor=bn.move_mean(abv,window=ma_len,min_count=1)/ma_len
        stock_data['fm106_'+str(int(ma_len))]=factor.copy()
        stock_data['fm106_'+str(int(ma_len))]=stock_data['fm106_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm106_'+str(int(ma_len)))
    return stock_data,factor_list

def fm107(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    ret_abs=ret.abs()
    ret_bak=ret*0
    for ma_len in [2**i for i in period_id_list]:
        ret_abs_quantile=ret_abs.rolling(ma_len).quantile(0.5)
        ret_bak.loc[ret>ret_abs_quantile]=ret.loc[ret>ret_abs_quantile]
        factor=ret_bak.rolling(ma_len).mean()
        stock_data['fm107_'+str(int(ma_len))]=factor.copy()
        stock_data['fm107_'+str(int(ma_len))]=stock_data['fm107_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm107_'+str(int(ma_len)))
    return stock_data,factor_list


def fm107_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    ret_abs=ret.abs()
    ret_bak=ret*0
    for ma_len in [2**i for i in period_id_list]:
        ret_abs_quantile=ret_abs.rolling(ma_len).quantile(0.5)
        ret_bak.loc[ret<-ret_abs_quantile]=ret.loc[ret<-ret_abs_quantile]
        factor=ret_bak.rolling(ma_len).mean()
        stock_data['fm107_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm107_2_'+str(int(ma_len))]=stock_data['fm107_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm107_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm107_3(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    ret_abs=ret.abs()
    ret_bak_up=ret*0
    ret_bak_down=ret*0
    for ma_len in [2**i for i in period_id_list]:
        ret_abs_quantile=ret_abs.rolling(ma_len).quantile(0.5)
        ret_bak_down.loc[ret<-ret_abs_quantile]=ret.loc[ret<-ret_abs_quantile]
        ret_bak_up.loc[ret>ret_abs_quantile]=ret.loc[ret>ret_abs_quantile]
        factor=ret_bak_up.rolling(ma_len).mean()+ret_bak_down.rolling(ma_len).mean()
        stock_data['fm107_3_'+str(int(ma_len))]=factor.copy()
        stock_data['fm107_3_'+str(int(ma_len))]=stock_data['fm107_3_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm107_3_'+str(int(ma_len)))
    return stock_data,factor_list

def fm108(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    ret_bak=ret*0
    for ma_len in [2**i for i in period_id_list]:
        vol_quantile=vol.rolling(ma_len).quantile(0.5)
        ret_bak.loc[vol>vol_quantile]=ret.loc[vol>vol_quantile]
        factor=ret_bak.rolling(ma_len).mean()
        stock_data['fm108_'+str(int(ma_len))]=factor.copy()
        stock_data['fm108_'+str(int(ma_len))]=stock_data['fm108_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm108_'+str(int(ma_len)))
    return stock_data,factor_list

def fm110(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    deltah=cl.diff()
    deltal=-op.diff()
    pdm=deltah*0
    mdm=deltal*0
    pdm.loc[deltah>deltal]=deltah[deltah>deltal]
    mdm.loc[deltah<deltal]=deltal[deltah<deltal]
    for ma_len in [2**i for i in period_id_list]:
        pdm_ma=pdm.rolling(ma_len).mean()
        mdm_ma=mdm.rolling(ma_len).mean()
        fenzi=(pdm_ma+mdm_ma)
        factor=(pdm_ma-mdm_ma)/fenzi
        factor.loc[np.abs(fenzi)<1e-5]=0
        stock_data['fm110_'+str(int(ma_len))]=factor.copy()
        stock_data['fm110_'+str(int(ma_len))]=stock_data['fm110_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm110_'+str(int(ma_len)))
    return stock_data,factor_list

def fm112(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    adv=0*op
    dec=0*op
    adv.loc[ret>0]=1
    dec.loc[ret<0]=-1
    advvol = adv*vol
    decvol = dec*vol
    for ma_len in [2**i for i in period_id_list]:
        advsum=adv.rolling(ma_len).sum()
        decsum=dec.rolling(ma_len).sum()
        advvolsum=advvol.rolling(ma_len).sum()
        decvolsum=decvol.rolling(ma_len).sum()
        fenzi=decsum*advvolsum
        factor=advsum*decvolsum/fenzi
        factor.loc[np.abs(fenzi)<1e-5]=0
        stock_data['fm112_'+str(int(ma_len))]=factor.copy()
        stock_data['fm112_'+str(int(ma_len))]=stock_data['fm112_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm112_'+str(int(ma_len)))
    return stock_data,factor_list

def fm113(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    vol_diff=vol.diff()
    adv=0*op
    dec=0*op
    adv.loc[vol_diff>0]=1
    dec.loc[vol_diff<0]=-1
    advret = adv*ret
    decret = dec*ret
    for ma_len in [2**i for i in period_id_list]:
        advsum=adv.rolling(ma_len).sum()
        decsum=dec.rolling(ma_len).sum()
        advretsum=advret.rolling(ma_len).sum()
        decretsum=decret.rolling(ma_len).sum()
        fenzi=decsum*advretsum
        factor=advsum*decretsum/fenzi
        factor.loc[np.abs(fenzi)<1e-5]=0
        stock_data['fm113_'+str(int(ma_len))]=factor.copy()
        stock_data['fm113_'+str(int(ma_len))]=stock_data['fm113_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm113_'+str(int(ma_len)))
    return stock_data,factor_list

def fm114(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    
    for ma_len in [2**i for i in period_id_list]:
        vwap_ret=vwap/vwap.shift(ma_len)-1
        vpt=vwap_ret*vol 
        factor=bn.move_rank(vpt,window=ma_len,min_count=1)
        stock_data['fm114_'+str(int(ma_len))]=factor.copy()
        stock_data['fm114_'+str(int(ma_len))]=stock_data['fm114_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm114_'+str(int(ma_len)))
    return stock_data,factor_list

def fm114_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        vwap_ret=vwap/vwap.shift(ma_len)-1
        vpt=vwap_ret
        factor=bn.move_rank(vpt,window=ma_len,min_count=1)
        stock_data['fm114_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm114_2_'+str(int(ma_len))]=stock_data['fm114_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm114_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm116(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    cl_diff=cl.diff()
    up=op*0
    down=op*0
    up.loc[cl_diff>0]=cl_diff[cl_diff>0]
    down.loc[cl_diff<0]=-cl_diff[cl_diff<0]
    for ma_len in [2**i for i in period_id_list]:
        up_sum=up.rolling(ma_len).mean()
        down_sum=down.rolling(ma_len).mean()
        fenzi=up_sum+down_sum
        factor=(up_sum-down_sum)/fenzi
        factor.loc[np.abs(fenzi)<1e-5]=0
        stock_data['fm116_'+str(int(ma_len))]=factor.copy()
        stock_data['fm116_'+str(int(ma_len))]=stock_data['fm116_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm116_'+str(int(ma_len)))
    return stock_data,factor_list

def fm117(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    hpc=high-cl.shift(1)
    pcl=cl.shift(1)-low
    for ma_len in [2**i for i in period_id_list]:
        hpc_mean=hpc.rolling(ma_len).mean()
        pcl_mean=pcl.rolling(ma_len).mean()
        fenzi=pcl_mean
        factor=hpc_mean/fenzi
        factor.loc[np.abs(fenzi)<1e-5]=0
        stock_data['fm117_'+str(int(ma_len))]=factor.copy()
        stock_data['fm117_'+str(int(ma_len))]=stock_data['fm117_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm117_'+str(int(ma_len)))
    return stock_data,factor_list

def fm118(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    premid=((high+low)/2).shift(1)
    hpc=high-premid
    pcl=premid-low
    for ma_len in [2**i for i in period_id_list]:
        hpc_mean=hpc.rolling(ma_len).mean()
        pcl_mean=pcl.rolling(ma_len).mean()
        fenzi=pcl_mean
        factor=hpc_mean/fenzi
        factor.loc[np.abs(fenzi)<1e-5]=0
        stock_data['fm118_'+str(int(ma_len))]=factor.copy()
        stock_data['fm118_'+str(int(ma_len))]=stock_data['fm118_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm118_'+str(int(ma_len)))
    return stock_data,factor_list

def fm122(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    pre_vwap=stock_data.vwap.shift(1)
    for ma_len in [2**i for i in period_id_list]:
        sizeup=0*op
        sizedown=0*op
        numup=0*op
        numdown=0*op
        ret=(cl-op)/op
        sizeup.loc[vwap>pre_vwap]=ret.loc[vwap>pre_vwap]
        sizedown.loc[vwap<pre_vwap]=-ret.loc[vwap<pre_vwap]
        numup.loc[vwap>pre_vwap]=1
        numdown.loc[vwap<pre_vwap]=1
        sizeup_sum=sizeup.rolling(ma_len,min_periods=1).sum()
        numup_sum=numup.rolling(ma_len,min_periods=1).sum()
        sizedown_sum=sizedown.rolling(ma_len,min_periods=1).sum()
        numdown_sum=numdown.rolling(ma_len,min_periods=1).sum()
        qbang=(sizeup_sum*numup_sum-sizedown_sum*numdown_sum)/(sizeup_sum*numup_sum+sizedown_sum*numdown_sum)
        fenmu=(sizeup_sum*numup_sum+sizedown_sum*numdown_sum)
        qbang.loc[fenmu.abs()<1e-9]=0
        stock_data['fm122_'+str(int(ma_len))]=qbang
        stock_data['fm122_'+str(int(ma_len))]=stock_data['fm122_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm122_'+str(int(ma_len)))
    return stock_data,factor_list

def fm124(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    pre_vwap=stock_data.vwap.shift(1)
    for ma_len in [2**i for i in period_id_list]:
        sizeup=0*op
        sizedown=0*op
        ret=(vwap-pre_vwap)/pre_vwap
        sizeup.loc[vwap>pre_vwap]=ret.loc[vwap>pre_vwap]
        sizedown.loc[vwap<pre_vwap]=-ret.loc[vwap<pre_vwap]
        sizeup_sum=sizeup.rolling(ma_len,min_periods=1).sum()
        sizedown_sum=sizedown.rolling(ma_len,min_periods=1).sum()
        qbang=(sizeup_sum-sizedown_sum)/(sizeup_sum+sizedown_sum)
        fenmu=(sizeup_sum+sizedown_sum)
        qbang.loc[fenmu.abs()<1e-9]=0
        stock_data['fm124_'+str(int(ma_len))]=qbang
        stock_data['fm124_'+str(int(ma_len))]=stock_data['fm124_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm124_'+str(int(ma_len)))
    return stock_data,factor_list

def fm128(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    add=0*op
    minus=0*op
    intra_ret=cl/op-1
    add.loc[intra_ret>0.001 ]=intra_ret.loc[intra_ret>0.001]
    minus.loc[intra_ret<-0.001]=intra_ret.loc[intra_ret<-0.001]
    tvi=add-minus
    for ma_len in [2**i for i in period_id_list]:
        factor=tvi.rolling(ma_len).mean()
        stock_data['fm128_'+str(int(ma_len))]=factor.copy()
        stock_data['fm128_'+str(int(ma_len))]=stock_data['fm128_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm128_'+str(int(ma_len)))
    return stock_data,factor_list

def fm128_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    add=0*op
    minus=0*op
    intra_ret=cl/op-1
    add.loc[intra_ret>0.001 ]=vol.loc[intra_ret>0.001]
    minus.loc[intra_ret<-0.001]=vol.loc[intra_ret<-0.001]
    tvi=(add-minus)/(add+minus)
    tvi.loc[(add+minus)<1e-3]=0
    for ma_len in [2**i for i in period_id_list]:
        factor=tvi.rolling(ma_len).mean()
        stock_data['fm128_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm128_2_'+str(int(ma_len))]=stock_data['fm128_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm128_2_'+str(int(ma_len)))
    return stock_data,factor_list


def fm128_3(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    add=0*op
    minus=0*op
    intra_ret=cl/op-1
    add.loc[intra_ret>0.003]=vol.loc[intra_ret>0.003]
    minus.loc[intra_ret<-0.003]=vol.loc[intra_ret<-0.003]
    tvi=(add-minus)/(add+minus)
    tvi.loc[(add+minus)<1e-3]=0
    for ma_len in [2**i for i in period_id_list]:
        factor=tvi.rolling(ma_len).mean()
        stock_data['fm128_3_'+str(int(ma_len))]=factor.copy()
        stock_data['fm128_3_'+str(int(ma_len))]=stock_data['fm128_3_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm128_3_'+str(int(ma_len)))
    return stock_data,factor_list

def fm128_4(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    add=0*op
    minus=0*op
    intra_ret=cl/op-1
    add.loc[intra_ret>0.005]=vol.loc[intra_ret>0.005]
    minus.loc[intra_ret<-0.005]=vol.loc[intra_ret<-0.005]
    tvi=(add-minus)/(add+minus)
    tvi.loc[(add+minus)<1e-3]=0
    for ma_len in [2**i for i in period_id_list]:
        factor=tvi.rolling(ma_len).mean()
        stock_data['fm128_4_'+str(int(ma_len))]=factor.copy()
        stock_data['fm128_4_'+str(int(ma_len))]=stock_data['fm128_4_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm128_4_'+str(int(ma_len)))
    return stock_data,factor_list

def fm129(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vhl=2*vwap-high-low
    hl=high-low
    for ma_len in [2**i for i in period_id_list]:
        ama=vhl.rolling(ma_len).mean()
        bma=hl.rolling(ma_len).mean()
        factor=ama/bma
        factor.loc[np.abs(bma)<1e-5]=0
        stock_data['fm129_'+str(int(ma_len))]=factor.copy()
        stock_data['fm129_'+str(int(ma_len))]=stock_data['fm129_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm129_'+str(int(ma_len)))
    return stock_data,factor_list

def fm132(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    delta=0*vwap
    delta.loc[vwap>vwap.shift(1)]=1
    delta.loc[vwap<vwap.shift(1)]=-1
    sign_vol=delta*vol
    ado=sign_vol*(vwap-low)/(high-low)
    ado.loc[np.abs(high-low)<1e-5]=0
    for ma_len in [2**i for i in period_id_list]:
        factor=ado.rolling(ma_len).mean()
        stock_data['fm132_'+str(int(ma_len))]=factor.copy()
        stock_data['fm132_'+str(int(ma_len))]=stock_data['fm132_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm132_'+str(int(ma_len)))
    return stock_data,factor_list

def fm133(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    delta=0*vwap
    delta.loc[vwap>vwap.shift(1)]=1
    delta.loc[vwap<vwap.shift(1)]=-1
    sign_vol=delta*vol
    ado=sign_vol*(cl-low)/(high-low)
    ado.loc[np.abs(high-low)<1e-5]=0
    for ma_len in [2**i for i in period_id_list]:
        factor=ado.rolling(ma_len).mean()
        stock_data['fm133_'+str(int(ma_len))]=factor.copy()
        stock_data['fm133_'+str(int(ma_len))]=stock_data['fm133_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm133_'+str(int(ma_len)))
    return stock_data,factor_list

def fm136(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    tmpval=vol*(2*cl-low-high)/(high-low)
    tmpval.loc[np.abs(high-low)<1e-5]=0
    for ma_len in [2**i for i in period_id_list]:
        tmpsum=tmpval.rolling(ma_len).mean()
        volsum=vol.rolling(ma_len).mean()
        factor=tmpsum/volsum
        factor.loc[np.abs(volsum)<1e-5]=0
        stock_data['fm136_'+str(int(ma_len))]=factor.copy()
        stock_data['fm136_'+str(int(ma_len))]=stock_data['fm136_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm136_'+str(int(ma_len)))
    return stock_data,factor_list

def fm136_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    tmpval=(2*cl-low-high)/(high-low)
    tmpval.loc[np.abs(high-low)<1e-5]=0
    for ma_len in [2**i for i in period_id_list]:
        tmpma=tmpval.rolling(ma_len).mean()
        factor=tmpma
        stock_data['fm136_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm136_2_'+str(int(ma_len))]=stock_data['fm136_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm136_2_'+str(int(ma_len)))
    return stock_data,factor_list


def fm137(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    volret=vol/vol.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        sumret=ret.rolling(ma_len).sum()
        sumvolret=volret.rolling(ma_len).sum()
        factor=sumret*sumvolret
        stock_data['fm137_'+str(int(ma_len))]=factor.copy()
        stock_data['fm137_'+str(int(ma_len))]=stock_data['fm137_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm137_'+str(int(ma_len)))
    return stock_data,factor_list

def fm138(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    volret=vol/vol.shift(1)-1
    dbret=ret*volret
    for ma_len in [2**i for i in period_id_list]:
        factor=dbret.rolling(ma_len).mean()
        stock_data['fm138_'+str(int(ma_len))]=factor.copy()
        stock_data['fm138_'+str(int(ma_len))]=stock_data['fm138_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm138_'+str(int(ma_len)))
    return stock_data,factor_list

def fm139(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    volret=vol/vol.shift(1)-1
    dbret=ret+volret
    for ma_len in [2**i for i in period_id_list]:
        factor=dbret.rolling(ma_len).mean()
        stock_data['fm139_'+str(int(ma_len))]=factor.copy()
        stock_data['fm139_'+str(int(ma_len))]=stock_data['fm139_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm139_'+str(int(ma_len)))
    return stock_data,factor_list

def fm141(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    un=op*0
    dn=op*0 
    un.loc[cl>op]=1
    dn.loc[cl<op]=-1
    udn=un+dn
    for ma_len in [2**i for i in period_id_list]:
        if ma_len==1:
            short_len=1
        else:
            short_len=int(ma_len/2)
        sma=udn.rolling(short_len).mean()
        lma=udn.rolling(ma_len).mean()
        factor=sma-lma
        stock_data['fm141_'+str(int(ma_len))]=factor.copy()
        stock_data['fm141_'+str(int(ma_len))]=stock_data['fm141_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm141_'+str(int(ma_len)))
    return stock_data,factor_list

def fm143(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        homean=(high-op).rolling(ma_len).mean()
        olmean=(op-low).rolling(ma_len).mean()
        factor=homean/olmean
        factor.loc[np.abs(olmean)<1e-5]=0
        stock_data['fm143_'+str(int(ma_len))]=factor.copy()
        stock_data['fm143_'+str(int(ma_len))]=stock_data['fm143_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm143_'+str(int(ma_len)))
    return stock_data,factor_list

def fm144(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_max(ret,window=ma_len,min_count=1)
        stock_data['fm144_'+str(int(ma_len))]=factor.copy()
        stock_data['fm144_'+str(int(ma_len))]=stock_data['fm144_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm144_'+str(int(ma_len)))
    return stock_data,factor_list

def fm144_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        if ma_len==1:
            period_len=1
        else:
            period_len=int(ma_len/2)
        ret_ma=ret.rolling(period_len).mean()
        factor=bn.move_max(ret_ma,window=ma_len,min_count=1)
        stock_data['fm144_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm144_2_'+str(int(ma_len))]=stock_data['fm144_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm144_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm144_3(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        if ma_len==1:
            period_len=1
        else:
            period_len=int(ma_len/2)
        ret_ma=ret.rolling(period_len).mean()
        factor=bn.move_min(ret_ma,window=ma_len,min_count=1)
        stock_data['fm144_3_'+str(int(ma_len))]=factor.copy()
        stock_data['fm144_3_'+str(int(ma_len))]=stock_data['fm144_3_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm144_3_'+str(int(ma_len)))
    return stock_data,factor_list

def fm144_4(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        if ma_len==1:
            period_len=1
        else:
            period_len=int(ma_len/2)
        ret_ma=ret.rolling(period_len).mean()
        factor=bn.move_max(ret_ma,window=ma_len,min_count=1)-bn.move_min(ret_ma,window=ma_len,min_count=1)
        stock_data['fm144_4_'+str(int(ma_len))]=factor.copy()
        stock_data['fm144_4_'+str(int(ma_len))]=stock_data['fm144_4_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm144_4_'+str(int(ma_len)))
    return stock_data,factor_list

def fm144_5(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_min(ret,window=ma_len,min_count=1)
        stock_data['fm144_5_'+str(int(ma_len))]=factor.copy()
        stock_data['fm144_5_'+str(int(ma_len))]=stock_data['fm144_5_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm144_5_'+str(int(ma_len)))
    return stock_data,factor_list

def fm144_6(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_max(ret,window=ma_len,min_count=1)-bn.move_min(ret,window=ma_len,min_count=1)
        stock_data['fm144_6_'+str(int(ma_len))]=factor.copy()
        stock_data['fm144_6_'+str(int(ma_len))]=stock_data['fm144_6_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm144_6_'+str(int(ma_len)))
    return stock_data,factor_list

def fm145(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    ret=cl/cl.shift(1)-1
    abs_ret=ret.abs()
    for ma_len in [2**i for i in period_id_list]:
        ret_ma=ret.rolling(int(ma_len)).mean()
        abs_ret_ma=abs_ret.rolling(ma_len).mean()
        factor=ret_ma/abs_ret_ma
        factor.loc[np.abs(abs_ret_ma)<1e-8]=0
        stock_data['fm145_'+str(int(ma_len))]=factor.copy()
        stock_data['fm145_'+str(int(ma_len))]=stock_data['fm145_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm145_'+str(int(ma_len)))
    return stock_data,factor_list

def fm146(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    vwap=stock_data.vwap
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    sign=np.sign(ret)
    sign_amount=sign*amount
    for ma_len in [2**i for i in period_id_list]:
        signamount_ma=sign_amount.rolling(ma_len).mean()
        amount_ma=amount.rolling(ma_len).mean()
        factor=signamount_ma/amount_ma
        factor.loc[np.abs(amount_ma)<1e-8]=0
        stock_data['fm146_'+str(int(ma_len))]=factor.copy()
        stock_data['fm146_'+str(int(ma_len))]=stock_data['fm146_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm146_'+str(int(ma_len)))
    return stock_data,factor_list

def fm150(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    ret_vol=np.abs(ret)/np.sqrt(vol)
    vwap_bak=0*vwap
    for ma_len in [2**i for i in period_id_list]:
        vwapma=vwap.rolling(ma_len).mean()
        retvol_quantile=pd.Series(ret_vol).rolling(ma_len).quantile(0.5)
        vwap_bak.loc[ret_vol>retvol_quantile]=vwap.loc[ret_vol>retvol_quantile]
        factor=vwap_bak.rolling(ma_len).mean()/vwapma
        stock_data['fm150_'+str(int(ma_len))]=factor.copy()
        stock_data['fm150_'+str(int(ma_len))]=stock_data['fm150_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm150_'+str(int(ma_len)))
    return stock_data,factor_list

def fm161(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    vwap_bak=0*vwap
    for ma_len in [2**i for i in period_id_list]:
        vwapma=vwap.rolling(ma_len).mean()
        amount_quantile=pd.Series(amount).rolling(ma_len).quantile(0.8)
        vwap_bak.loc[amount>amount_quantile]=vwap.loc[amount>amount_quantile]
        factor=vwap_bak.rolling(ma_len).mean()/vwapma
        stock_data['fm161_'+str(int(ma_len))]=factor.copy()
        stock_data['fm161_'+str(int(ma_len))]=stock_data['fm161_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm161_'+str(int(ma_len)))
    return stock_data,factor_list

def fm162(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    ret_vol=np.abs(ret)/amount
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_rank(ret_vol,window=ma_len,min_count=1)
        stock_data['fm162_'+str(int(ma_len))]=factor.copy()
        stock_data['fm162_'+str(int(ma_len))]=stock_data['fm162_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm162_'+str(int(ma_len)))
    return stock_data,factor_list

def fm163(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=np.array(stock_data.vol)
    amount=stock_data.amount
    ret=np.array(cl/cl.shift(1)-1)
    ret_vol=np.abs(ret)/amount
    
    ret_up = np.zeros(len(stock_data))
    vol_up = np.zeros(len(stock_data))
    ret_dw = np.zeros(len(stock_data))
    vol_dw = np.zeros(len(stock_data))
    ret_up[ret>0] = ret[ret>0]
    vol_up[ret>0]=vol[ret>0]
    ret_dw[ret<0] = ret[ret<0]
    vol_dw[ret<0]=vol[ret<0]
    illq_up = abs(ret_up)/(np.sqrt(vol_up)+0.0001)
    illq_up[vol_up==0] = 0
    illq_up[np.isnan(illq_up)] = 0
    illq_dw = abs(ret_dw)/(np.sqrt(vol_dw)+0.0001)
    illq_dw[vol_dw==0] = 0
    illq_dw[np.isnan(illq_dw)] = 0
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_mean(illq_up,window=ma_len,min_count=1)-bn.move_mean(illq_dw,window=ma_len,min_count=1)
        stock_data['fm163_'+str(int(ma_len))]=factor.copy()
        stock_data['fm163_'+str(int(ma_len))]=stock_data['fm163_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm163_'+str(int(ma_len)))
    return stock_data,factor_list

def fm164(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=np.array(stock_data.vol)
    amount=stock_data.amount
    ret=np.array(cl/cl.shift(1)-1)
    ret_vol=np.abs(ret)/amount
    
    ret_up = np.zeros(len(stock_data))
    ret_dw = np.zeros(len(stock_data))
    ret_up[ret>0] = ret[ret>0]
    ret_dw[ret<0] = ret[ret<0]
    ret_up_2=ret_up**2
    ret_dw_2=ret_dw**2
  
    for ma_len in [2**i for i in period_id_list]:
        factor=bn.move_mean(ret_up_2,window=ma_len,min_count=1)-bn.move_mean(ret_dw_2,window=ma_len,min_count=1)
        stock_data['fm164_'+str(int(ma_len))]=factor.copy()
        stock_data['fm164_'+str(int(ma_len))]=stock_data['fm164_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm164_'+str(int(ma_len)))
    return stock_data,factor_list


def fm165(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    pvt=ret*vol
    for ma_len in [2**i for i in period_id_list]:
        pvt_sum=pvt.rolling(ma_len).sum()
        factor=bn.move_rank(pvt_sum,window=ma_len,min_count=1)
        stock_data['fm165_'+str(int(ma_len))]=factor.copy()
        stock_data['fm165_'+str(int(ma_len))]=stock_data['fm165_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm165_'+str(int(ma_len)))
    return stock_data,factor_list

def fm172(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
  
    for ma_len in [2**i for i in period_id_list]:
        vol_ma=vol.rolling(ma_len).mean()
        vol_std_short=vol_ma.rolling(ma_len).std()
        vol_std_long=vol_ma.rolling(ma_len*2).std()
        factor=vol_std_short/vol_std_long 
        factor.loc[np.abs(vol_std_long)<1e-3]=0
        stock_data['fm172_'+str(int(ma_len))]=factor.copy()
        stock_data['fm172_'+str(int(ma_len))]=stock_data['fm172_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm172_'+str(int(ma_len)))
    return stock_data,factor_list

def fm173(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    ret_p2=ret**2
    ret_p3=ret**3
    for ma_len in [2**i for i in period_id_list]:
        ret_p2_ma=ret_p2.rolling(ma_len).mean()
        ret_p3_ma=ret_p3.rolling(ma_len).mean()
        factor=ret_p3_ma/ret_p2_ma**1.5
        factor.loc[np.abs(ret_p2_ma**1.5)<1e-9]=0
        stock_data['fm173_'+str(int(ma_len))]=factor.copy()
        stock_data['fm173_'+str(int(ma_len))]=stock_data['fm173_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm173_'+str(int(ma_len)))
    return stock_data,factor_list

def fm174(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    ret_p2=ret**2
    ret_p3=ret**4
    for ma_len in [2**i for i in period_id_list]:
        ret_p2_ma=ret_p2.rolling(ma_len).mean()
        ret_p3_ma=ret_p3.rolling(ma_len).mean()
        factor=ret_p3_ma/ret_p2_ma**2
        factor.loc[np.abs(ret_p2_ma**2)<1e-9]=0
        stock_data['fm174_'+str(int(ma_len))]=factor.copy()
        stock_data['fm174_'+str(int(ma_len))]=stock_data['fm174_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm174_'+str(int(ma_len)))
    return stock_data,factor_list

def fm179(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        vpct=vol/vol.rolling(ma_len).mean()
        factor=vpct.rolling(ma_len).mean()
        stock_data['fm179_'+str(int(ma_len))]=factor.copy()
        stock_data['fm179_'+str(int(ma_len))]=stock_data['fm179_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm179_'+str(int(ma_len)))
    return stock_data,factor_list

def fm181(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    factor=np.zeros(len(stock_data))
    for ma_len in [2**i for i in period_id_list]:
        factor[:]=0
        for i in range(ma_len,len(stock_data)):
            vol_wind=vol.values[(i-ma_len+1):(i+1)]
            cls_wind=cl.values[(i-ma_len+1):(i+1)]
            cls_last=cls_wind[-1]
            factor[i]=np.sum(vol_wind[cls_wind>cls_last])/np.sum(vol_wind)
        stock_data['fm181_'+str(int(ma_len))]=factor.copy()
        stock_data['fm181_'+str(int(ma_len))]=stock_data['fm181_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm181_'+str(int(ma_len)))
    return stock_data,factor_list

def fm184(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    
    for ma_len in [2**i for i in period_id_list]:
        sumret=ret.rolling(ma_len).sum()
        factor=bn.move_max(sumret,window=ma_len,min_count=1)
        stock_data['fm184_'+str(int(ma_len))]=factor.copy()
        stock_data['fm184_'+str(int(ma_len))]=stock_data['fm184_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm184_'+str(int(ma_len)))
    return stock_data,factor_list

def fm184_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    
    for ma_len in [2**i for i in period_id_list]:
        sumret=ret.rolling(ma_len).sum()
        factor=bn.move_min(sumret,window=ma_len,min_count=1)
        stock_data['fm184_2_'+str(int(ma_len))]=factor.copy()
        stock_data['fm184_2_'+str(int(ma_len))]=stock_data['fm184_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm184_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm184_3(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    
    for ma_len in [2**i for i in period_id_list]:
        sumret=ret.rolling(ma_len).sum()
        factor=bn.move_max(sumret,window=ma_len,min_count=1)-bn.move_min(sumret,window=ma_len,min_count=1)
        stock_data['fm184_3_'+str(int(ma_len))]=factor.copy()
        stock_data['fm184_3_'+str(int(ma_len))]=stock_data['fm184_3_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm184_3_'+str(int(ma_len)))
    return stock_data,factor_list


def fm185(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    pos=0*ret
    neg=0*ret
    pos.loc[ret>0]=1
    neg.loc[ret<0]=1
    for ma_len in [2**i for i in period_id_list]:
        factor=pos.rolling(ma_len).mean()-neg.rolling(ma_len).mean()
        stock_data['fm185_'+str(int(ma_len))]=factor.copy()
        stock_data['fm185_'+str(int(ma_len))]=stock_data['fm185_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm185_'+str(int(ma_len)))
    return stock_data,factor_list

def fm187(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    factor=np.zeros(len(stock_data))
    ret_bak=0*ret
    for ma_len in [2**i for i in period_id_list]:
        amount_percentile=amount.rolling(ma_len).quantile(0.8)
        ret_bak.loc[amount>amount_percentile]=ret.loc[amount>amount_percentile]      
        stock_data['fm187_'+str(int(ma_len))]=ret_bak.rolling(ma_len).mean()
        stock_data['fm187_'+str(int(ma_len))]=stock_data['fm187_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm187_'+str(int(ma_len)))
    return stock_data,factor_list

def fm187_2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    factor=np.zeros(len(stock_data))
    ret_bak=0*ret
    for ma_len in [2**i for i in period_id_list]:
        amount_percentile=amount.rolling(ma_len).quantile(0.2)
        ret_bak.loc[amount<amount_percentile]=ret.loc[amount<amount_percentile]      
        stock_data['fm187_2_'+str(int(ma_len))]=ret_bak.rolling(ma_len).mean()
        stock_data['fm187_2_'+str(int(ma_len))]=stock_data['fm187_2_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm187_2_'+str(int(ma_len)))
    return stock_data,factor_list

def fm187_3(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    factor=np.zeros(len(stock_data))
    ret_bak_80=0*ret
    ret_bak_20=0*ret
    for ma_len in [2**i for i in period_id_list]:
        amount_percentile_80=amount.rolling(ma_len).quantile(0.8)
        ret_bak_80.loc[amount>amount_percentile_80]=ret.loc[amount>amount_percentile_80] 
        amount_percentile_20=amount.rolling(ma_len).quantile(0.2)
        ret_bak_20.loc[amount<amount_percentile_20]=ret.loc[amount<amount_percentile_20]
        stock_data['fm187_3_'+str(int(ma_len))]=ret_bak_80.rolling(ma_len).mean()-ret_bak_20.rolling(ma_len).mean()
        stock_data['fm187_3_'+str(int(ma_len))]=stock_data['fm187_3_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm187_3_'+str(int(ma_len)))
    return stock_data,factor_list

def fm188(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    pre_cl=cl.shift(1)
    tl=np.minimum(low,pre_cl)
    th=np.maximum(high,pre_cl)
    diff=np.zeros(len(stock_data))
    diff[cl>pre_cl]=cl[cl>pre_cl]-tl[cl>pre_cl]
    diff[cl<pre_cl]=cl[cl<pre_cl]-th[cl<pre_cl]
    for ma_len in [2**i for i in period_id_list]:
        acd=bn.move_mean(diff,window=ma_len,min_count=1)
        clsma=cl.rolling(ma_len).mean()
        factor=acd/clsma
        stock_data['fm188_'+str(int(ma_len))]=factor.copy()
        stock_data['fm188_'+str(int(ma_len))]=stock_data['fm188_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm188_'+str(int(ma_len)))
    return stock_data,factor_list


def fm189(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    pre_cl=cl.shift(1)
    pre_op=op.shift(1)
    hmo=high-op
    ompo=op-pre_op 
    oml=op-low 
    pomo=pre_op-op
    max1=np.maximum(hmo,ompo)
    max2=np.maximum(oml,pomo)
    dtm=np.zeros(len(stock_data))
    dtm[op>pre_op]=max1[op>pre_op]
    dbm=np.zeros(len(stock_data))
    dbm[op<pre_op]=max2[op<pre_op]
    dtm=pd.Series(dtm)
    dbm=pd.Series(dbm)

    for ma_len in [2**i for i in period_id_list]:
        stm=dtm.rolling(ma_len).sum()
        sbm=dbm.rolling(ma_len).sum()
        r1=(stm-sbm)/stm 
        r2=(stm-sbm)/sbm
        factor=r1
        factor.loc[stm<sbm]=r2.loc[stm<sbm]
        stock_data['fm189_'+str(int(ma_len))]=factor.copy()
        stock_data['fm189_'+str(int(ma_len))]=stock_data['fm189_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm189_'+str(int(ma_len)))
    return stock_data,factor_list

def fm190(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.close
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    pre_cl=cl.shift(1)
    pre_op=op.shift(1)
    hmo=high-op
    ompo=op-pre_op 
    oml=op-low 
    pomo=pre_op-op
    max1=np.maximum(hmo,ompo)
    max2=np.maximum(oml,pomo)
    dtm=np.zeros(len(stock_data))
    dtm[op>pre_op]=max1[op>pre_op]
    dbm=np.zeros(len(stock_data))
    dbm[op<pre_op]=max2[op<pre_op]
    dtm=pd.Series(dtm)
    dbm=pd.Series(dbm)

    for ma_len in [2**i for i in period_id_list]:
        stm=dtm.rolling(ma_len).sum()
        sbm=dbm.rolling(ma_len).sum()
        r1=(stm-sbm)/stm 
        r2=(stm-sbm)/sbm
        factor=r1
        factor.loc[stm<sbm]=r2.loc[stm<sbm]
        stock_data['fm190_'+str(int(ma_len))]=factor.copy()
        stock_data['fm190_'+str(int(ma_len))]=stock_data['fm190_'+str(int(ma_len))].fillna(0)
        factor_list.append('fm190_'+str(int(ma_len)))
    return stock_data,factor_list

def fm191(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    
    pre_op=op.shift(1)
    pre_high=high.shift(1)
    pre_low=low.shift(1)
    pre_cl=cl.shift(1)
    hmpc=high-pre_cl  
    pcml=pre_cl-low 
    max1=hmpc*0
    max2=pcml*0 
    max1.loc[hmpc>0]=hmpc.loc[hmpc>0]
    max2.loc[pcml>0]=pcml.loc[pcml>0]
    for ma_len in [2**i for i in period_id_list]:
        denom=max1.rolling(ma_len).sum()
        numer=max2.rolling(ma_len).sum()
        factor=denom/numer 
        factor.loc[np.abs(numer)<1e-5]=0
        stock_data['fm191_'+str(int(ma_len))]=factor.copy()
        stock_data['fm191_'+str(int(ma_len))]=stock_data['fm191_'+str(int(ma_len))].fillna(1)
        factor_list.append('fm191_'+str(int(ma_len)))
    return stock_data,factor_list

def fm192(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    op=stock_data.open
    vol=stock_data.vol
    amount=stock_data.amount
    ret=cl/cl.shift(1)-1
    hl=(high)/low
    for ma_len in [2**i for i in period_id_list]:
        factor=hl.rolling(ma_len).mean()
        stock_data['fm192_'+str(int(ma_len))]=factor.copy()
        stock_data['fm192_'+str(int(ma_len))]=stock_data['fm192_'+str(int(ma_len))].fillna(1)
        factor_list.append('fm192_'+str(int(ma_len)))
    return stock_data,factor_list

def dev(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        factor=(cp-cp.rolling(ma_len).mean())/cp.rolling(ma_len).mean()
        factor.loc[factor.abs()>100]=0
        
        stock_data['dev_'+str(int(ma_len))]=factor
        stock_data['dev_'+str(int(ma_len))]=stock_data['dev_'+str(int(ma_len))].fillna(0)
        factor_list.append('dev_'+str(int(ma_len)))
    return stock_data,factor_list

def rawret(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vol=stock_data['vol']
    cp_ret=cp/cp.shift(1)-1
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        stock_data['rawret_'+str(int(ma_len))]=cp_ret.rolling(ma_len).sum()
        stock_data['rawret_'+str(int(ma_len))]=stock_data['rawret_'+str(int(ma_len))].fillna(0)
        factor_list.append('rawret_'+str(int(ma_len)))
    return stock_data,factor_list

def rwi(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    for ma_len in [2**i for i in period_id_list]:
        hl=stock_data['high']-stock_data['low']
        hc=(stock_data['high']-stock_data['close'].shift(1)).abs()
        lc=(stock_data['low']-stock_data['close'].shift(1)).abs()
        tr=hl
        tr.loc[tr<hc]=hc.loc[tr<hc]
        tr.loc[tr<lc]=lc.loc[tr<lc]
        atr=tr.rolling(ma_len,min_periods=1).mean()
        rwih=(high-low.rolling(ma_len,min_periods=1).min())/atr
        rwil=(-low+high.rolling(ma_len,min_periods=1).max())/atr
        rwi=rwih-rwil
        rwi.loc[atr.abs()<1e-9]=0
        stock_data['rwi_'+str(int(ma_len))]=rwi
        factor_list.append('rwi_'+str(int(ma_len)))
        stock_data['rwi_'+str(int(ma_len))]=stock_data['rwi_'+str(int(ma_len))].fillna(0)
    return stock_data,factor_list

def dpo(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        ma_short=cl.rolling(ma_len).mean()
        ma_long=cl.rolling(ma_len*2).mean()
        factor=(ma_short-ma_long)/ma_long        
        stock_data['dpo_'+str(int(ma_len))]=factor
        stock_data['dpo_'+str(int(ma_len))]=stock_data['dpo_'+str(int(ma_len))].fillna(0)
        factor_list.append('dpo_'+str(int(ma_len)))
    return stock_data,factor_list

def m9(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cp=stock_data.close
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        highmax = high.rolling(ma_len,min_periods=1).max()
        lowmin = low.rolling(ma_len,min_periods=1).min()
        mid = (highmax + lowmin) / 2
        vwapmean =vwap.rolling(ma_len,min_periods=1).mean()
        m9=(vwapmean - mid) / vwapmean
        m9.loc[vwapmean.abs()<1e-9]=0
        stock_data['m9_'+str(int(ma_len))]=m9
        stock_data['m9_'+str(int(ma_len))]=stock_data['m9_'+str(int(ma_len))].fillna(0)
        factor_list.append('m9_'+str(int(ma_len)))
    return stock_data,factor_list

def cv(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    vol=stock_data['vol']
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        meanprod=stock_data['close'].rolling(ma_len).mean()*stock_data['vol'].rolling(ma_len).mean()
        prodmean=(stock_data['close']*stock_data['vol']).rolling(ma_len).mean()
        cv=prodmean/meanprod
        cv.loc[meanprod.abs()<1e-9]=0
        stock_data['cv_'+str(int(ma_len))]=cv.fillna(0)
        factor_list.append('cv_'+str(int(ma_len)))
    return stock_data,factor_list

def cr(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        premid=(high.shift(1)+low.shift(1))/2
        hm=high-premid
        ml=premid-low
        hmsum=hm.rolling(ma_len,min_periods=1).sum()
        mlsum=ml.rolling(ma_len,min_periods=1).sum()
        cr=hmsum/mlsum
        cr.loc[mlsum.abs()<1e-5]=0
        stock_data['cr_'+str(int(ma_len))]=cr
        factor_list.append('cr_'+str(int(ma_len)))
        stock_data['cr_'+str(int(ma_len))]=stock_data['cr_'+str(int(ma_len))].fillna(0)
    return stock_data,factor_list

def rsi(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        lc=stock_data['close']-stock_data['close'].shift(1)
        lc_new=stock_data['close']-stock_data['close'].shift(1)
        alc=lc_new.abs()
        lc_mean=lc.rolling(ma_len,min_periods=1).mean()
        alc_mean=alc.rolling(ma_len,min_periods=1).mean()
        rsi=lc_mean/alc_mean
        rsi.loc[alc_mean.abs()<1e-9]=0
        stock_data['rsi_'+str(int(ma_len))]=rsi
        stock_data['rsi_'+str(int(ma_len))]=stock_data['rsi_'+str(int(ma_len))].fillna(0)
        factor_list.append('rsi_'+str(int(ma_len)))
    return stock_data,factor_list

def cmo(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    op=stock_data['open']
    cp=stock_data['close']
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        diff=cp.diff()
        su=cp*0
        sd=cp*0
        su.loc[cp>cp.shift(1)]=diff.loc[cp>cp.shift(1)]
        sd.loc[cp<cp.shift(1)]=-diff.loc[cp<cp.shift(1)]
        su_sum=su.rolling(ma_len,min_periods=1).sum()
        sd_sum=sd.rolling(ma_len,min_periods=1).sum()
        cmo=(su_sum-sd_sum)/(su_sum+sd_sum)
        fenmu=(su_sum+sd_sum)
        cmo.loc[fenmu.abs()<1e-9]=0
        stock_data['cmo_'+str(int(ma_len))]=cmo
        factor_list.append('cmo_'+str(int(ma_len)))
        stock_data['cmo_'+str(int(ma_len))]=stock_data['cmo_'+str(int(ma_len))].fillna(0)
    return stock_data,factor_list  

def whlr(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        hh=stock_data['high'].rolling(ma_len).max()
        ll=stock_data['low'].rolling(ma_len).min()
        prehh=hh.shift(ma_len)
        prell=ll.shift(ma_len)
        stock_data['whlr_'+str(int(ma_len))]=(hh/prell+ll/prehh)/2
        stock_data['whlr_'+str(int(ma_len))]=stock_data['whlr_'+str(int(ma_len))].fillna(0)
        factor_list.append('whlr_'+str(int(ma_len)))
    return stock_data,factor_list

def bop(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    co=cl-op
    hl=high-low
    co_hl=co/hl
    co_hl.loc[hl.abs()<1e-5]=0
    for ma_len in [2**i for i in period_id_list]:
        factor=co_hl.rolling(ma_len).mean()
        stock_data['bop_'+str(int(ma_len))]=factor
        factor_list.append('bop_'+str(int(ma_len)))
        stock_data['bop_'+str(int(ma_len))]=stock_data['bop_'+str(int(ma_len))].fillna(0)
    return stock_data,factor_list


def real(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        highmax=high.rolling(ma_len).max()
        lowmin=low.rolling(ma_len).min()
        ret=(cp-cp.shift(ma_len))
        temp=ret/(highmax-lowmin)
        temp.loc[(highmax-lowmin)<1e-5]=0
        stock_data['real_'+str(int(ma_len))]=temp
        stock_data['real_'+str(int(ma_len))]=stock_data['real_'+str(int(ma_len))].fillna(0)
        factor_list.append('real_'+str(int(ma_len)))
    return stock_data,factor_list

def dis(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        highmax=high.rolling(ma_len).max()
        lowmin=low.rolling(ma_len).min()
        mid=(highmax+lowmin)/2
        temp=(cp-mid)/(highmax-lowmin)
        temp.loc[(highmax-lowmin)<1e-5]=0
        stock_data['dis_'+str(int(ma_len))]=temp
        stock_data['dis_'+str(int(ma_len))]=stock_data['dis_'+str(int(ma_len))].fillna(0)
        factor_list.append('dis_'+str(int(ma_len)))
    return stock_data,factor_list

def sharp(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        ret=stock_data['close'].diff()/stock_data['close'].shift(1)
        ret_mean=ret.rolling(ma_len,min_periods=1).mean()
        ret_std=ret.rolling(ma_len,min_periods=1).std()
        sharp=ret_mean/ret_std
        sharp.loc[ret_std.abs()<1e-9]=0
        stock_data['sharp_'+str(int(ma_len))]=sharp
        stock_data['sharp_'+str(int(ma_len))]=stock_data['sharp_'+str(int(ma_len))].fillna(0)
        factor_list.append('sharp_'+str(int(ma_len)))
    return stock_data,factor_list

def ddi(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    hl=high+low
    high_abs=(high.diff()).abs()
    low_abs=(low.diff()).abs()
    dmz=0*high
    dmf=0*high
    dmz.loc[hl>hl.shift(1)]=np.maximum(high_abs,low_abs)
    dmf.loc[hl<hl.shift(1)]=np.maximum(high_abs,low_abs)
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        diz=dmz.rolling(ma_len).sum()
        dif=dmf.rolling(ma_len).sum()
        fenmu=diz+dif
        factor=(diz-dif)/fenmu
        factor.loc[fenmu.abs()<1e-5]=0
        stock_data['ddi_'+str(int(ma_len))]=factor
        stock_data['ddi_'+str(int(ma_len))]=stock_data['ddi_'+str(int(ma_len))].fillna(0)
        factor_list.append('ddi_'+str(int(ma_len)))
    return stock_data,factor_list

def auad(stock_data,period_id_list):
    high=np.array(stock_data.high)
    low=np.array(stock_data.low)
    vwap=stock_data.vwap
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        h_dis=ma_len-1-bn.move_argmax(high,window=ma_len,min_count=1)
        l_dis=ma_len-1-bn.move_argmin(low,window=ma_len,min_count=1)
        au=(ma_len-h_dis)/ma_len
        ad=(ma_len-l_dis)/ma_len
        auad=(au-ad)/(au+ad)
        stock_data['auad_'+str(int(ma_len))]=auad
        stock_data['au_'+str(int(ma_len))]=au
        stock_data['ad_'+str(int(ma_len))]=ad
        stock_data['auad_'+str(int(ma_len))]=stock_data['auad_'+str(int(ma_len))].fillna(0)
        stock_data['au_'+str(int(ma_len))]=stock_data['au_'+str(int(ma_len))].fillna(0)
        stock_data['ad_'+str(int(ma_len))]=stock_data['ad_'+str(int(ma_len))].fillna(0)
        factor_list.append('auad_'+str(int(ma_len)))
        factor_list.append('au_'+str(int(ma_len)))
        factor_list.append('ad_'+str(int(ma_len)))
    return stock_data,factor_list

def qbang(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    for ma_len in [2**i for i in period_id_list]:
        sizeup=0*op
        sizedown=0*op
        numup=0*op
        numdown=0*op
        ret=(cl-op)/op
        sizeup.loc[cl>op]=ret.loc[cl>op]
        sizedown.loc[cl<op]=-ret.loc[cl<op]
        numup.loc[cl>op]=1
        numdown.loc[cl<op]=1
        sizeup_sum=sizeup.rolling(ma_len,min_periods=1).sum()
        numup_sum=numup.rolling(ma_len,min_periods=1).sum()
        sizedown_sum=sizedown.rolling(ma_len,min_periods=1).sum()
        numdown_sum=numdown.rolling(ma_len,min_periods=1).sum()
        qbang=(sizeup_sum*numup_sum-sizedown_sum*numdown_sum)/(sizeup_sum*numup_sum+sizedown_sum*numdown_sum)
        fenmu=(sizeup_sum*numup_sum+sizedown_sum*numdown_sum)
        qbang.loc[fenmu.abs()<1e-9]=0
        stock_data['qbang_'+str(int(ma_len))]=qbang
        stock_data['qbang_'+str(int(ma_len))]=stock_data['qbang_'+str(int(ma_len))].fillna(0)
        factor_list.append('qbang_'+str(int(ma_len)))
    return stock_data,factor_list

def rank(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=np.array(stock_data.close)
    vwap=stock_data.vwap
    vol=stock_data['vol']
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        close_rank=bn.move_rank(cp,window=ma_len,min_count=1)
        stock_data['rank_'+str(int(ma_len))]=close_rank
        stock_data['rank_'+str(int(ma_len))]=stock_data['rank_'+str(int(ma_len))].fillna(0)
        factor_list.append('rank_'+str(int(ma_len)))
    return stock_data,factor_list

def hhv(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    vol=stock_data['vol']
    factor_list=[]
    dh=stock_data.high
    dl=stock_data.low
    dv=stock_data.vol
    dvwap=stock_data.vwap
    for ma_len in [2**i for i in period_id_list]:
        hh=dh.rolling(ma_len).max()
        dv1=(dvwap*dv).rolling(ma_len).mean()/dv.rolling(ma_len).mean()
        hhv=dv1/hh
        hhv.loc[hh.abs()<1e-9]=0
        hhv.loc[hhv.abs()>1000]=0
        stock_data['hhv_'+str(int(ma_len))]=hhv.fillna(0)
        factor_list.append('hhv_'+str(int(ma_len)))
    return stock_data,factor_list


def tii(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        close_ma=cl.rolling(ma_len).mean()
        dev=cl-close_ma
        dev_pos=0*cl
        dev_neg=0*cl
        dev_pos.loc[dev>0]=dev 
        dev_neg.loc[dev<0]=-dev
        sum_pos=dev_pos.rolling(ma_len).sum()
        sum_neg=dev_neg.rolling(ma_len).sum()
        fenmu=(sum_pos+sum_neg)
        factor=sum_pos/fenmu
        factor.loc[fenmu.abs()<1e-4]=0
        stock_data['tii_'+str(int(ma_len))]=factor
        stock_data['tii_'+str(int(ma_len))]=stock_data['tii_'+str(int(ma_len))].fillna(0)
        factor_list.append('tii_'+str(int(ma_len)))
    return stock_data,factor_list


def dmi(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    pdm=high.diff()
    mdm=-low.diff()
    pdm.loc[pdm<0]=0
    mdm.loc[mdm<0]=0
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        pdi=pdm.rolling(ma_len).sum()
        mdi=mdm.rolling(ma_len).sum()
        fenmu=pdi+mdi
        dmi=(pdi-mdi)/(pdi+mdi)
        dmi.loc[fenmu.abs()<0.00001]=0
        stock_data['dmi_'+str(int(ma_len))]=dmi
        stock_data['dmi_'+str(int(ma_len))]=stock_data['dmi_'+str(int(ma_len))].fillna(0)
        factor_list.append('dmi_'+str(int(ma_len)))
    return stock_data,factor_list

def wvad(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    op=stock_data.open
    vol=stock_data.vol
    factor_list=[]
    cl=stock_data['close']
    a=cl-op
    b=stock_data.high-stock_data.low
    ab=a/b
    ab.loc[b.abs()<1e-9]=0
    for ma_len in [2**i for i in period_id_list]:
        wvad=(ab*vol).rolling(ma_len,min_periods=1).mean()/vol.rolling(ma_len,min_periods=1).mean()
        fenmu=vol.rolling(ma_len,min_periods=1).mean()
        wvad.loc[fenmu.abs()<1e-9]=0
        stock_data['wvad_'+str(int(ma_len))]=wvad
        factor_list.append('wvad_'+str(int(ma_len)))
        stock_data['wvad_'+str(int(ma_len))]=stock_data['wvad_'+str(int(ma_len))].fillna(0)
    return stock_data,factor_list


def macd(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cp=stock_data.close
    factor_list=[]
    ret=cp/cp.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        dif=ret.rolling(ma_len).mean()-ret.rolling(2*ma_len).mean()
        dea=dif.rolling(ma_len).mean()
        macd=dif-dea
        stock_data['macd_'+str(int(ma_len))]=macd
        stock_data['macd_'+str(int(ma_len))]=stock_data['macd_'+str(int(ma_len))].fillna(0)
        factor_list.append('macd_'+str(int(ma_len)))
    return stock_data,factor_list

def adx(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    factor_list=[]
    hl=stock_data['high']-stock_data['low']
    hc=(stock_data['high']-stock_data['close'].shift(1)).abs()
    lc=(stock_data['low']-stock_data['close'].shift(1)).abs()
    tr=hl
    tr.loc[tr<hc]=hc.loc[tr<hc]
    tr.loc[tr<lc]=lc.loc[tr<lc]
    for ma_len in [2**i for i in period_id_list]:
        tr=tr.rolling(ma_len).sum()   #ATR
        hd=stock_data['high']-stock_data['high'].shift(1)
        ld=-stock_data['low']+stock_data['low'].shift(1)
        dmp=hd.copy()
        dmp.loc[(hd<0)|(hd<ld)]=0
        dmp=dmp.rolling(ma_len).sum()
        dmm=ld.copy()
        dmm.loc[(ld<0)|(ld<hd)]=0
        dmm=dmm.rolling(ma_len).sum()
        adx=(dmp-dmm)/tr
        adx.loc[tr.abs()<1e-9]=0
        stock_data['adx_'+str(int(ma_len))]=adx.fillna(0)
        factor_list.append('adx_'+str(int(ma_len)))
    return stock_data,factor_list

def vhf(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        highmax=high.rolling(ma_len).max()
        lowmin=low.rolling(ma_len).min()
        diff_abs=cp.diff().abs()
        fenmu=diff_abs.rolling(ma_len).sum()
        vhf=(highmax-lowmin)/fenmu
        vhf.loc[fenmu.abs()<1e-9]=0
        stock_data['vhf_'+str(int(ma_len))]=vhf
        stock_data['vhf_'+str(int(ma_len))]=stock_data['vhf_'+str(int(ma_len))].fillna(0)
        factor_list.append('vhf_'+str(int(ma_len)))
    return stock_data,factor_list

def mfi(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        tp=(stock_data['high']+stock_data['low']+stock_data['close'])/3
        pre_tp=tp.shift(1)
        cf_add=tp*0
        cf_minus=tp*0
        cf_add.loc[tp>pre_tp]=tp.loc[tp>pre_tp]*vol.loc[tp>pre_tp]
        cf_minus.loc[tp<pre_tp]=tp.loc[tp<pre_tp]*vol.loc[tp<pre_tp]
        add_sum=cf_add.rolling(ma_len,min_periods=1).sum()
        minus_sum=cf_minus.rolling(ma_len,min_periods=1).sum()
        fenmu=(add_sum+minus_sum)
        mfi=(add_sum-minus_sum)/(add_sum+minus_sum)
        mfi.loc[fenmu.abs()<1e-9]=0
        stock_data['mfi_'+str(int(ma_len))]=mfi
        stock_data['mfi_'+str(int(ma_len))]=stock_data['mfi_'+str(int(ma_len))].fillna(0)
        factor_list.append('mfi_'+str(int(ma_len)))        
    return stock_data,factor_list      

def kdjd(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    for ma_len in [2**i for i in period_id_list]:
        highmax=high.rolling(ma_len).max()
        lowmin=low.rolling(ma_len).min()
        fenmu=highmax-lowmin
        stoch=(cl-lowmin)/(highmax-lowmin)
        stoch.loc[fenmu.abs()<1e-5]=0
        stoch_max=stoch.rolling(ma_len).max()
        stoch_min=stoch.rolling(ma_len).min()
        fenmu=(stoch_max-stoch_min)
        factor=(stoch-stoch_min)/fenmu
        factor.loc[fenmu.abs()<1e-5]=0
        stock_data['kdjd_'+str(int(ma_len))]=factor
        factor_list.append('kdjd_'+str(int(ma_len)))
        stock_data['kdjd_'+str(int(ma_len))]=stock_data['kdjd_'+str(int(ma_len))].fillna(0)
    return stock_data,factor_list


def pdi(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    pdm=high.diff()
    mdm=-low.diff()
    pdm.loc[pdm<0]=0
    mdm.loc[mdm<0]=0
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        hl=stock_data['high']-stock_data['low']
        hc=(stock_data['high']-stock_data['close'].shift(1)).abs()
        lc=(stock_data['low']-stock_data['close'].shift(1)).abs()
        tr=hl
        tr.loc[tr<hc]=hc.loc[tr<hc]
        tr.loc[tr<lc]=lc.loc[tr<lc]
        tr=tr.rolling(ma_len).sum()
        hd=stock_data['high']-stock_data['high'].shift(1)
        ld=-stock_data['low']+stock_data['low'].shift(1)
        dmp=hd.copy()
        dmp.loc[(hd<0)|(hd<ld)]=0
        dmp=dmp.rolling(ma_len).sum()
        dmm=ld.copy()
        dmm.loc[(ld<0)|(ld<hd)]=0
        pdi=dmp/tr
        pdi.loc[tr.abs()<1e-5]=0
        stock_data['pdi_'+str(int(ma_len))]=pdi
        factor_list.append('pdi_'+str(int(ma_len)))
        stock_data['pdi_'+str(int(ma_len))]=stock_data['pdi_'+str(int(ma_len))].fillna(0)
    return stock_data,factor_list

def boll(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        close_ma=cl.rolling(ma_len,min_periods=1).mean()
        close_std=cl.rolling(ma_len,min_periods=1).std()
        boll=(cl-close_ma)/close_std
        boll.loc[close_std.abs()<1e-9]=0
        stock_data['boll_'+str(int(ma_len))]=boll
        stock_data['boll_'+str(int(ma_len))]=stock_data['boll_'+str(int(ma_len))].fillna(0)
        factor_list.append('boll_'+str(int(ma_len)))
    return stock_data,factor_list

def adtm(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cl=stock_data.close
    factor_list=[]
    op=stock_data['open']
    cl=stock_data['close']
    dtm=0*op
    dbm=0*op
    op_dif=op.diff()
    dtm.loc[op_dif>0]=np.maximum(high-op,op_dif)
    dbm.loc[op_dif<0]=np.maximum(op-low,-op_dif)
    for ma_len in [2**i for i in period_id_list]:
        stm=dtm.rolling(ma_len).sum()
        sbm=dbm.rolling(ma_len).sum()
        fenmu=stm+sbm
        factor=(stm-sbm)/fenmu
        factor.loc[fenmu.abs()<1e-5]=0
        stock_data['adtm_'+str(int(ma_len))]=factor
        factor_list.append('adtm_'+str(int(ma_len)))
        stock_data['adtm_'+str(int(ma_len))]=stock_data['adtm_'+str(int(ma_len))].fillna(0)
    return stock_data,factor_list

def abi(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    op=stock_data['open']
    cl=stock_data['close']
    vol=stock_data.vol
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        a_boll=cl*0
        b_boll=cl*0
        a_boll.loc[cl>op]=1
        b_boll.loc[cl<op]=1
        a=a_boll.rolling(ma_len,min_periods=1).sum()
        b=b_boll.rolling(ma_len,min_periods=1).sum()
        abi=(a-b)/(a+b)        
        fenmu=a+b
        abi.loc[fenmu.abs()<1e-5]=0
        stock_data['abi_'+str(int(ma_len))]=abi
        factor_list.append('abi_'+str(int(ma_len)))
        stock_data['abi_'+str(int(ma_len))]=stock_data['abi_'+str(int(ma_len))].fillna(0)
    return stock_data,factor_list   

def obv(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    vol=stock_data.vol
    factor_list=[]
    dire_vol=vol.copy()
    dire_vol.loc[cp<=cp.shift(1)]=-dire_vol.loc[cp<=cp.shift(1)]
    for ma_len in [2**i for i in period_id_list]:
        temp=dire_vol.rolling(int(ma_len)).mean()/vol.rolling(ma_len).mean()
        temp.loc[vol.rolling(ma_len).mean()<1e-9]=0
        stock_data['obv_'+str(int(ma_len))]=temp
        stock_data['obv_'+str(int(ma_len))]=stock_data['obv_'+str(int(ma_len))].fillna(0)
        factor_list.append('obv_'+str(int(ma_len)))
    return stock_data,factor_list

def up(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    factor_list=[]
    up_num=cp*0
    up_num.loc[cp>cp.shift(1)]=1
    for ma_len in [2**i for i in period_id_list]:
        stock_data['up_'+str(int(ma_len))]=up_num.rolling(ma_len).mean()
        stock_data['up_'+str(int(ma_len))]=stock_data['up_'+str(int(ma_len))].fillna(0)
        factor_list.append('up_'+str(int(ma_len)))
    return stock_data,factor_list

def adjsharp(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    ret=cl/cl.shift(1)-1
    ret_up=0*cl
    ret_down=0*cl
    ret_up.loc[ret>0]=ret.loc[ret>0]
    ret_down.loc[ret<=0]=ret.loc[ret<=0]
    for ma_len in [2**i for i in period_id_list]:
        up_mean=ret_up.rolling(ma_len).mean()
        up_std=ret_up.rolling(ma_len).std()
        down_mean=ret_down.rolling(ma_len).mean()
        down_std=ret_down.rolling(ma_len).std()
        up_sharp=up_mean/up_std
        up_sharp.loc[np.abs(up_std)<1e-10]=0
        down_sharp=down_mean/down_std
        down_sharp.loc[np.abs(down_std)<1e-10]=0
        stock_data['adjsharp_'+str(int(ma_len))]=up_sharp+down_sharp
        stock_data['adjsharp_'+str(int(ma_len))]=stock_data['adjsharp_'+str(int(ma_len))].fillna(0)
        factor_list.append('adjsharp_'+str(int(ma_len)))
    return stock_data,factor_list

def cjb(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        cjb=cl/low.rolling(ma_len).min()-cl/cl.shift(ma_len)
        stock_data['cjb_'+str(int(ma_len))]=cjb
        stock_data['cjb_'+str(int(ma_len))]=stock_data['cjb_'+str(int(ma_len))].fillna(0)
        factor_list.append('cjb_'+str(int(ma_len)))
    return stock_data,factor_list

def htb(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        htb=cl/high.rolling(ma_len).max()-cl/cl.shift(ma_len)
        stock_data['htb_'+str(int(ma_len))]=htb
        stock_data['htb_'+str(int(ma_len))]=stock_data['htb_'+str(int(ma_len))].fillna(0)
        factor_list.append('htb_'+str(int(ma_len)))
    return stock_data,factor_list

def fluc(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        hh=stock_data['close'].rolling(ma_len).max()
        ll=stock_data['close'].rolling(ma_len).min()
        factor=(hh-ll)/(cl+cl.shift(ma_len))
        stock_data['fluc_'+str(int(ma_len))]=factor
        stock_data['fluc_'+str(int(ma_len))]=stock_data['fluc_'+str(int(ma_len))].fillna(0)
        factor_list.append('fluc_'+str(int(ma_len)))
    return stock_data,factor_list

def delta(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    amount=stock_data.amount
    vol=stock_data.vol
    factor_list=[]
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        amt_sum_long=amount.rolling(ma_len*2).mean()
        vol_sum_long=vol.rolling(ma_len*2).mean()
        amt_sum_short=amount.rolling(ma_len).mean()
        vol_sum_short=vol.rolling(ma_len).mean()
        vwap_short=amt_sum_short/vol_sum_short
        vwap_long=amt_sum_long/vol_sum_long
        vwap_short=vwap_short.fillna(method='ffill')
        vwap_long=vwap_long.fillna(method='ffill')
        delta=(vwap_short-vwap_long)/vwap_long
        stock_data['delta_'+str(int(ma_len))]=delta
        stock_data['delta_'+str(int(ma_len))]=stock_data['delta_'+str(int(ma_len))].fillna(0)
        factor_list.append('delta_'+str(int(ma_len)))
    return stock_data,factor_list


def dens(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    hl=high-low
    for ma_len in [2**i for i in period_id_list]:
        hh=stock_data['high'].rolling(ma_len).max()
        ll=stock_data['low'].rolling(ma_len).min()
        fenzi=hh-ll
        factor=hl.rolling(ma_len).sum()/(hh-ll)
        factor.loc[np.abs(fenzi)<1e-5]=0
        stock_data['dens_'+str(int(ma_len))]=factor
        stock_data['dens_'+str(int(ma_len))]=stock_data['dens_'+str(int(ma_len))].fillna(0)
        factor_list.append('dens_'+str(int(ma_len)))
    return stock_data,factor_list

def skew(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        ret_skew=ret.rolling(ma_len).skew()
        stock_data['skew_'+str(int(ma_len))]=ret_skew
        stock_data['skew_'+str(int(ma_len))]=stock_data['skew_'+str(int(ma_len))].fillna(0)
        factor_list.append('skew_'+str(int(ma_len)))
    return stock_data,factor_list

def std(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    ret=cl/cl.shift(1)-1
    for ma_len in [2**i for i in period_id_list]:
        ret_std=ret.rolling(ma_len).std()
        stock_data['std_'+str(int(ma_len))]=ret_std
        stock_data['std_'+str(int(ma_len))]=stock_data['std_'+str(int(ma_len))].fillna(0)
        factor_list.append('std_'+str(int(ma_len)))
    return stock_data,factor_list


def hl(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        hh=stock_data['high'].rolling(ma_len).max()
        ll=stock_data['low'].rolling(ma_len).min()
        prehh=stock_data['high'].rolling(2*ma_len).max()
        prell=stock_data['low'].rolling(2*ma_len).min()
        stock_data['hl_'+str(int(ma_len))]=(hh/prell+ll/prehh)/2
        stock_data['hl_'+str(int(ma_len))]=stock_data['hl_'+str(int(ma_len))].fillna(0)
        factor_list.append('hl_'+str(int(ma_len)))
    return stock_data,factor_list

def hl2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        hh=stock_data['high'].rolling(ma_len).max()
        ll=stock_data['low'].rolling(ma_len).min()
        prehh=stock_data['high'].rolling(2*ma_len).max()
        prell=stock_data['low'].rolling(2*ma_len).min()
        stock_data['hl2_'+str(int(ma_len))]=(hh/prehh+ll/prell)/2
        stock_data['hl2_'+str(int(ma_len))]=stock_data['hl2_'+str(int(ma_len))].fillna(0)
        factor_list.append('hl2_'+str(int(ma_len)))
    return stock_data,factor_list

def hl3(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        hh=stock_data['high'].rolling(ma_len).max()
        ll=stock_data['low'].rolling(ma_len).min()
        prehh=hh.shift(2*ma_len)
        prell=ll.shift(2*ma_len)
        stock_data['hl3_'+str(int(ma_len))]=(hh/prehh+ll/prell)/2
        stock_data['hl3_'+str(int(ma_len))]=stock_data['hl3_'+str(int(ma_len))].fillna(0)
        factor_list.append('hl3_'+str(int(ma_len)))
    return stock_data,factor_list


def hl4(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        hh=stock_data['high'].rolling(ma_len).max()
        ll=stock_data['low'].rolling(ma_len).min()
        prehh=hh.shift(2*ma_len)
        prell=ll.shift(2*ma_len)
        stock_data['hl4_'+str(int(ma_len))]=(hh/prell+ll/prehh)/2
        stock_data['hl4_'+str(int(ma_len))]=stock_data['hl4_'+str(int(ma_len))].fillna(0)
        factor_list.append('hl4_'+str(int(ma_len)))
    return stock_data,factor_list

def ch(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    ret=cp/cp.shift(1)-1
    vol=stock_data.vol
    amt=stock_data.amount
    factor_list=[]
    cph=cp/high
    for ma_len in [2**i for i in period_id_list]:
        factor=cph.rolling(ma_len).mean()
        stock_data['ch_'+str(int(ma_len))]=factor.copy()
        stock_data['ch_'+str(int(ma_len))]=stock_data['ch_'+str(int(ma_len))].fillna(0)
        factor_list.append('ch_'+str(int(ma_len)))
    return stock_data,factor_list

def cl(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    ret=cp/cp.shift(1)-1
    vol=stock_data.vol
    amt=stock_data.amount
    factor_list=[]
    cpl=cp/low
    for ma_len in [2**i for i in period_id_list]:
        factor=cpl.rolling(ma_len).mean()
        stock_data['cl_'+str(int(ma_len))]=factor.copy()
        stock_data['cl_'+str(int(ma_len))]=stock_data['cl_'+str(int(ma_len))].fillna(0)
        factor_list.append('cl_'+str(int(ma_len)))
    return stock_data,factor_list

def lj(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    hl=high-low
    vol=stock_data.vol
    for ma_len in [2**i for i in period_id_list]:
        hlv1=hl.rolling(ma_len).mean()*vol.rolling(ma_len).mean()
        hlv2=(hl*vol).rolling(ma_len).mean()
        fz=hlv1+hlv2
        factor=(hlv1-hlv2)/fz
        factor.loc[np.abs(fz)<1e-5]=0
        stock_data['lj_'+str(int(ma_len))]=factor
        stock_data['lj_'+str(int(ma_len))]=stock_data['lj_'+str(int(ma_len))].fillna(0)
        factor_list.append('lj_'+str(int(ma_len)))
    return stock_data,factor_list

def upvol(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    ret=cl/cl.shift(1)-1
    ret_square=ret**2
    for ma_len in [2**i for i in period_id_list]:
        up_ret_square=0*ret_square
        up_ret_square.loc[ret>0]=ret_square.loc[ret>0]
        fenzi=up_ret_square.rolling(ma_len).mean()
        fenmu=ret_square.rolling(ma_len).mean()
        factor=fenzi/fenmu
        factor.loc[np.abs(fenmu)<1e-9]=0
        stock_data['upvol_'+str(int(ma_len))]=factor
        stock_data['upvol_'+str(int(ma_len))]=stock_data['upvol_'+str(int(ma_len))].fillna(0)
        factor_list.append('upvol_'+str(int(ma_len)))
    return stock_data,factor_list

def upvol2(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    ret=cl/cl.shift(1)-1
    ret_square=ret**2
    for ma_len in [2**i for i in period_id_list]:
        ret_mean=ret.rolling(ma_len).mean()
        up_ret_square=0*ret_square
        up_ret_square.loc[ret>ret_mean]=ret_square.loc[ret>ret_mean]
        fenzi=up_ret_square.rolling(ma_len).mean()
        fenmu=ret_square.rolling(ma_len).mean()
        factor=fenzi/fenmu
        factor.loc[np.abs(fenmu)<1e-9]=0
        stock_data['upvol2_'+str(int(ma_len))]=factor
        stock_data['upvol2_'+str(int(ma_len))]=stock_data['upvol2_'+str(int(ma_len))].fillna(0)
        factor_list.append('upvol2_'+str(int(ma_len)))
    return stock_data,factor_list

def vskew(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        factor=vwap.rolling(ma_len).skew()
        stock_data['vskew_'+str(int(ma_len))]=factor.copy()
        stock_data['vskew_'+str(int(ma_len))]=stock_data['vskew_'+str(int(ma_len))].fillna(0)
        factor_list.append('vskew_'+str(int(ma_len)))
    return stock_data,factor_list

def active(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    cp=stock_data.close
    vwap=stock_data.vwap
    active_buy=stock_data.active_volume
    active_sell=stock_data.vol-stock_data.active_volume
    factor_list=[]
    for ma_len in [2**i for i in period_id_list]:
        factor=(active_buy-active_sell)/(active_buy+active_sell)
        fenmu=(active_buy+active_sell)
        factor.loc[np.abs(fenmu)<1e-5]=0
        factor=factor.rolling(ma_len).mean()
        factor.loc[factor.abs()>100]=0
        stock_data['active_'+str(int(ma_len))]=factor
        stock_data['active_'+str(int(ma_len))]=stock_data['active_'+str(int(ma_len))].fillna(0)
        factor_list.append('active_'+str(int(ma_len)))
    return stock_data,factor_list

def retabs(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    ret=cl/cl.shift(1)
    for ma_len in [2**i for i in period_id_list]:
        ret_abs=ret.rolling(ma_len).mean().abs()
        stock_data['retabs_'+str(int(ma_len))]=ret_abs
        stock_data['retabs_'+str(int(ma_len))]=stock_data['retabs_'+str(int(ma_len))].fillna(0)
        factor_list.append('retabs_'+str(int(ma_len)))
    return stock_data,factor_list

def retstd(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    ret=cl/cl.shift(1)
    for ma_len in [2**i for i in period_id_list]:
        ret_std=ret.rolling(ma_len).std()
        stock_data['retstd_'+str(int(ma_len))]=ret_std
        stock_data['retstd_'+str(int(ma_len))]=stock_data['retstd_'+str(int(ma_len))].fillna(0)
        factor_list.append('retstd_'+str(int(ma_len)))
    return stock_data,factor_list

def trades(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    vol=stock_data.vol
    actb=stock_data.active_volume
    acts=vol-stock_data.active_volume
    factor_list=[]
    ret=cl/cl.shift(1)
    trade_num=stock_data.trades
    for ma_len in [2**i for i in period_id_list]:
        factor=trade_num.rolling(ma_len).mean()
        stock_data['trades_'+str(int(ma_len))]=factor
        stock_data['trades_'+str(int(ma_len))]=stock_data['trades_'+str(int(ma_len))].fillna(0)
        factor_list.append('trades_'+str(int(ma_len)))
    return stock_data,factor_list

def volstd(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    vol=stock_data.vol
    for ma_len in [2**i for i in period_id_list]:
        vol_std=vol.rolling(ma_len).std()
        vol_ma=vol.rolling(ma_len).mean()
        factor=vol_std/vol_ma
        factor[np.abs(vol_ma)<1e-5]=0
        stock_data['volstd_'+str(int(ma_len))]=factor
        stock_data['volstd_'+str(int(ma_len))]=stock_data['volstd_'+str(int(ma_len))].fillna(0)
        factor_list.append('volstd_'+str(int(ma_len)))
    return stock_data,factor_list

def volskew(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    vol=stock_data.vol
    for ma_len in [2**i for i in period_id_list]:
        volskew=vol.rolling(ma_len).skew()
        stock_data['volskew_'+str(int(ma_len))]=volskew
        stock_data['volskew_'+str(int(ma_len))]=stock_data['volskew_'+str(int(ma_len))].fillna(0)
        factor_list.append('volskew_'+str(int(ma_len)))
    return stock_data,factor_list

def volratio(stock_data,period_id_list):
    high=stock_data.high
    low=stock_data.low
    vwap=stock_data.vwap
    cl=stock_data.close
    factor_list=[]
    vol=stock_data.vol
    for ma_len in [2**i for i in period_id_list]:
        vol_ma=vol.rolling(ma_len).mean()
        pre_vol=vol_ma.shift(2*ma_len)
        factor=vol_ma/pre_vol 
        factor[np.abs(pre_vol)<1e-5]=0
        stock_data['volratio_'+str(int(ma_len))]=factor
        stock_data['volratio_'+str(int(ma_len))]=stock_data['volratio_'+str(int(ma_len))].fillna(0)
        factor_list.append('volratio_'+str(int(ma_len)))
    return stock_data,factor_list


func_list=['fm2','fm2_2','fm2_3','fm2_4','fm11','fm12','fm12_2','fm14','fm18','fm21','fm21_2','fm21_3','fm25','fm25_2','fm25_3','fm25_4','fm25_5','fm25_6','fm25_7','fm32',
'fm33','fm33_2','fm34','fm49','fm61','fm62','fm65','fm66','fm69','fm70','fm71','fm77','fm78','fm79','fm80','fm81','fm82','fm83','fm87','fm93','fm102',
'fm104','fm104_2','fm106','fm107','fm107_2','fm107_3','fm108','fm110','fm112','fm113','fm114','fm114_2','fm116','fm117','fm118','fm122','fm124','fm128','fm128_2','fm128_3',
'fm128_4','fm129','fm132','fm133','fm136','fm136_2','fm137','fm138','fm139','fm141','fm143','fm144','fm144_2','fm144_3','fm144_4','fm144_5','fm144_6','fm145','fm146','fm150',
'fm161','fm162','fm163','fm164','fm165','fm172','fm173','fm174','fm179','fm184','fm184_2','fm184_3','fm185','fm187','fm187_2','fm187_3','fm188','fm189','fm190','fm191',
'fm192','dev','rawret','rwi','dpo','m9','cv','cr','rsi','cmo','whlr','bop','real','dis','sharp','ddi','auad','qbang','rank','hhv','tii','dmi','wvad','macd','adx','vhf','mfi',
'kdjd','pdi','boll','adtm','abi','obv','up','adjsharp','cjb','htb','delta','dens','skew','std','hl','hl2','hl3','hl4','fluc','ch','cl','lj','upvol','upvol2','vskew',
'active','retabs','retstd','trades','volstd','volskew','volratio']

