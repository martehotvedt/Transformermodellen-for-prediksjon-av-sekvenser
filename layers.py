import numpy as np
from utils import onehot

class Layer:

    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        
        return

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,grad):
        raise NotImplementedError
    
    def step_gd(self,alpha):
        """
        Performs a gradient descent step given learning rate.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {         
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                },
            'w2': {....},
            
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params:
            self.params[param]['w'] -= alpha*self.params[param]['d']
        return

    def adam_step(self, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.j += 1
        for param in self.params:
            G = self.params[param]['d']     # dL/dW_j
            M = self.params[param]['m']
            V = self.params[param]['v']

            # Lagrer M_j og V_j
            self.params[param]['m'] = beta1 * M + (1 - beta1) * G
            self.params[param]['v'] = beta2 * V + (1 - beta2) * (G*G)
            
            # Beregner M_hat og V_hat
            M_hat = 1 / (1 - beta1**self.j) * self.params[param]['m']        
            V_hat = 1 / (1 - beta2**self.j) * self.params[param]['v']
            
            # Beregner W_{j+1}: Oppdaterer matrisene
            self.params[param]['w'] -= alpha * (np.divide(M_hat, (np.sqrt(V_hat) + epsilon)))
        
        return    
  




class Attention(Layer):

    def __init__(self, d, k):
        # Definerer verdier som skal være tilgjengelig innenfor hele klassen:
        self.j = 0
        init_scale = 0.1
        self.softmax = Softmax()
        self.z = None                
        self.dLdz = None

        # Initialiserer vektene ved bruk av en normalfordeling, skalert med init_scale 
        # Initialiserer også flere underliggende matriser som skal brukes senere
        self.params = {"W_Q":{'w': np.random.randn(k,d) * init_scale, 'd': None, 'm': np.zeros((k,d)), 'v': np.zeros((k,d))},                   
                        "W_K":{'w': np.random.randn(k,d) * init_scale, 'd': None, 'm': np.zeros((k,d)), 'v': np.zeros((k,d))},
                        "W_O":{'w': np.random.randn(k,d) * init_scale, 'd': None, 'm': np.zeros((k,d)), 'v': np.zeros((k,d))},
                        "W_V":{'w': np.random.randn(k,d) * init_scale, 'd': None, 'm': np.zeros((k,d)), 'v': np.zeros((k,d))}}
        return

    def forward(self,z):

        # Lagrer z, samt dimensjonene
        self.z = z
        b, d, n = z.shape             

        # Definerer D-matrisa
        i1, i2 = np.tril_indices(n,-1)      
        D = np.zeros((n,n))
        D[i1,i2] -= np.inf

        arg_softmax = np.einsum("bdn, kd, kD, bDN -> bnN", z, self.params["W_Q"]['w'], self.params["W_K"]['w'], z, optimize = True) + D   # Argumentet som A tar inn
        self.A = self.softmax.forward(arg_softmax)

        z_l = z + np.einsum("kd, kD, bDN, bNn -> bdn", self.params["W_O"]['w'], self.params["W_V"]['w'], z, self.A, optimize = True)

        return z_l

    def backward(self,grad):
        """
        grad = g_l : derivert av loss-funksjonen mhp den endelige outputen fra forward-passet
        """

        g_OV = np.einsum("dk,kD,bDn -> bdn", np.transpose(self.params["W_V"]['w']), self.params["W_O"]['w'], grad, optimize = True)

        g_S = self.softmax.backward( np.einsum("bdn,bdN-> bnN", self.z, g_OV, optimize = True) )

        self.dLdz = ( grad + np.einsum("bdn,bNn->bdN", g_OV, self.A, optimize = True)               
                       + np.einsum("kd,kD,bDN,bNn-> bdn", self.params["W_K"]['w'], self.params["W_Q"]['w'], self.z, g_S, optimize = True) 
                       + np.einsum("kd,kD,bDn,bNn-> bdN", self.params["W_Q"]['w'], self.params["W_K"]['w'], self.z, g_S, optimize = True) )

        # Fyller inn for parametrene dL/dW_i, i = O, V, K, Q
        self.params["W_O"]['d'] = np.einsum("kd,bdn,bnN,bDN -> kD", self.params["W_V"]['w'], self.z, self.A, grad, optimize = True)
        self.params["W_V"]['d'] = np.einsum("kd,bdn,bNn,bDN -> kD", self.params["W_O"]['w'], grad, self.A, self.z, optimize = True) 
        self.params["W_K"]['d'] = np.einsum("kd,bdn,bnN,bDN -> kD", self.params["W_Q"]['w'], self.z, g_S, self.z, optimize = True)
        self.params["W_Q"]['d'] = np.einsum("kd,bdn,bNn,bDN -> kD", self.params["W_K"]['w'], self.z, g_S, self.z, optimize = True)
      
        return self.dLdz
    


class Softmax(Layer):
    """
    Sannsynlighetsfordelig
    """

    def __init__(self):   
        self.epsilon = 10**(-8)
        return 

    
    def forward(self, x):    #skal lage diskret sannsynlighet matrise av x
    
        self.P = np.exp(x-x.max(axis=1,keepdims=True))  #setter alle elementer lik e^elementet

        self.Q = np.sum(self.P, axis=1,keepdims=True)   #summer over hver kolonne slik at hvert element i den
                                                        #kolonnen har samme verdi lik summen
        self.z_l = np.divide(self.P, (self.Q + self.epsilon))   #deler hvert element på summen i den samme posisjonen

        return self.z_l


    def backward(self, grad):

        S = np.divide(self.P, ( self.Q * self.Q + self.epsilon))  # definerer variabelen S

        b, a, s = grad.shape          # henter dimensjonen til gradienten som tas inn
        new_grad = np.zeros_like(S)     # vil padde venste side av g_l
        new_grad[:, :, -s:] += grad       # gir g_l lik dimensjon som z_l ved å sette inn null-kolonne(r) fremst 

        col_sum = np.sum(np.multiply(new_grad, S), axis=1, keepdims=True)       #finner summen av hver kolonne og setter hvert element
                                                                                #i kolonnene lik summen av kolonnen
        deriverte = np.multiply(new_grad, self.z_l) - np.multiply(col_sum, self.P)      #regner ut den deriverte av objektfunksjonen med hensyn på z
                                                             
        return deriverte


class CrossEntropy(Layer):
    """
    Objektfunksjonen L
    """

    def __init__(self):
        self.epsilon = 10 **(-8)    # skal sørge for at vi ikke gjør operasjoner med null der det ikke er mulig
        return
        
    def forward(self,Z,y):

        self.b, self.m, self.n = Z.shape    # henter størrelsene på Z 
        k, self.i = y.shape                 # og y for å kunne bruke disse senere

        self.y = y
        self.Y_hat = Z[:,:,-self.i:]    # henter verdiene til Y_hat ved å slice Z, slik at Y_hat og y får like mange kolonner

        self.p = np.einsum("m,bmc->bc", np.ones(self.m), np.multiply(self.Y_hat, onehot(self.y, self.m)), optimize = True)    # 1^T * Y_hat * Y

        self.q = -np.log(self.p) 

        self.L = np.mean(self.q)       # beregner verdien til loss-funksjonen

        return self.L

    def backward(self):
        dLdY_hat = np.multiply((-1 / self.i), (np.divide(onehot(self.y,self.m), (self.Y_hat + self.epsilon) )) )  # Beregner dL/dY_hat

        dLdZ = np.zeros((self.b, self.m, self.n))     # padder med null-kolonner
        dLdZ[:,:, -dLdY_hat.shape[2]:] += dLdY_hat    # slik at gradienten får lik dimensjon som Z

        return dLdZ                                   # returnerer gradienten dL/dZ
    


class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self,input_size, output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.j = 0
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w), 'm':np.zeros_like(self.w), 'v':np.zeros_like(self.w)}}
        

    def forward(self,x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """
 
        self.x = x
        
        #Return output of layer
        #y = w@x
        y = np.einsum('od,bdn->bon',self.params['w']['w'],x, optimize = True)
        return y
        
    def backward(self,grad):
        """
        Performs backward pass. def __init__(self, d, k):

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """
        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt weight w: 
        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params['w']['d'] = np.einsum('bon,bdn->od',grad,self.x, optimize = True) * 1/b

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        return np.einsum('od,bon->bdn',self.params['w']['w'],grad, optimize = True)
    

class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        return

    def relu(self,x):
        #relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,x):
        
        #Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self,grad):

        #dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))



class EmbedPosition(Layer):
    def __init__(self,n_max,m,d,init_scale=1e-1):   

        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """
        self.j = 0

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max)*init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp":{'w':self.w,'d':None, 'm': np.zeros((d,n_max)), 'v': np.zeros((d,n_max))}}

    def forward(self,X):

        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n). 
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to 

        z_0 = W_E@X + W_P[:,:n]

        """

        #We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params['Wp']['w'][:,:n]
        return z_0
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params['Wp']['d'] = np.zeros_like(self.w)
        self.params['Wp']['d'] += np.sum(grad,axis=0)/b

        #Use backwards pass of the linear layer
        self.embed.backward(grad)

        #This is always the final layer, so we return None
        return None
    
    def step_gd(self,step_size):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_gd(step_size)

    def adam_step(self, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.embed.adam_step()
        super().adam_step()



class FeedForward(Layer):


    def __init__(self,d, p,init_scale = 0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,init_scale)

        #We use the Relu activation function
        self.activation = Relu()
        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,init_scale)


    def forward(self,x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)
        
        """
        #We use backward pass of the linear layers and activation.
        #Recall that the backward pass reverse the order of the layers.
        a = self.l2.backward(grad) 
        c = self.activation.backward(a)
        grad_feed_forward = self.l1.backward(c)

        #Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward


    def step_gd(self,step_size):

        #Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)

    def adam_step(self, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.l1.adam_step()
        self.l2.adam_step()