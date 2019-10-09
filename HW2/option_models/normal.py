# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:56:58 2017

@author: jaehyuk
"""
import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt

def normal_formula(strike, spot, vol, texp, intr=0.0, divr=0.0, cp_sign=1):
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac

    if( texp<0 or vol*np.sqrt(texp)<1e-8 ):
        return disc_fac * np.fmax( cp_sign*(forward-strike), 0 )

    vol_std = np.fmax(vol * np.sqrt(texp), 1.0e-16)
    d = (forward - strike) / vol_std

    price = disc_fac * (cp_sign * (forward - strike) * ss.norm.cdf(cp_sign * d) + vol_std * ss.norm.pdf(d))
    return price

class NormalModel:
    
    vol, intr, divr = None, None, None
    
    def __init__(self, vol, intr=0, divr=0):
        self.vol = vol
        self.intr = intr
        self.divr = divr
    
    def price(self, strike, spot, texp, cp_sign=1):
        return normal_formula(strike, spot, self.vol, texp, intr=self.intr, divr=self.divr, cp_sign=cp_sign)
    
    def dcal(self, strike, spot, texp):
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac * div_fac

        vol_std = np.fmax(self.vol * np.sqrt(texp), 1.0e-16)
        d = (forward - strike) / vol_std
        return d
    
    def delta(self, strike, spot, texp, cp_sign=1):
        ''' 
        <-- PUT your implementation here
        '''
        d = self.dcal(strike, spot, texp)
        
        return cp_sign*ss.norm.cdf(cp_sign*d)

    def vega(self, strike, spot, texp, cp_sign=1):
        ''' 
        <-- PUT your implementation here
        '''
        d = self.dcal(strike, spot, texp)
        
        return ss.norm.pdf(d) * np.sqrt(texp)

    def gamma(self, strike, spot, texp, cp_sign=1):
        ''' 
        <-- PUT your implementation here
        '''
        d = self.dcal(strike, spot, texp)
        
        return ss.norm.pdf(d) / (self.vol*np.sqrt(texp))

    def impvol(self, price, strike, spot, texp, cp_sign=1):
        ''' 
        <-- PUT your implementation here
        '''
        iv_func = lambda _vol: \
            normal_formula(strike, spot, _vol, texp, self.intr, self.divr, cp_sign) - price
        vol = sopt.brentq(iv_func, 0, spot)
        return vol