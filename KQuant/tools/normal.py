import math

class Normal :

    """
    Normal distribution (PDF and CDF).
    """
     
    @staticmethod
    def pdf(x, mu, sigma) : 
        # Gaussian density N(mu, sigma^2)
        return math.exp(-((x-mu)**2)/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)
    
    @staticmethod
    def cdf(x, mu, sigma) :
        # Standardization to N(0,1) then use erf
        z = (x-mu)/sigma
        return (1+math.erf(z/math.sqrt(2)))/2
    