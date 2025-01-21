import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module

class Linear_quadratic(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear_quadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = Parameter(torch.Tensor(out_features, in_features))
        self.weight_g = Parameter(torch.Tensor(out_features, in_features))
        self.weight_b = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = bias
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_g = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_g', None)
            self.register_parameter('bias_b', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_r, a=0)
        init.kaiming_uniform_(self.weight_g, a=0)
        init.kaiming_uniform_(self.weight_b, a=0)
        if self.bias is not None:
            fan_in_r, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            fan_in_g, _ = init._calculate_fan_in_and_fan_out(self.weight_g)
            fan_in_b, _ = init._calculate_fan_in_and_fan_out(self.weight_b)
            bound_r = 1 / math.sqrt(fan_in_r)
            bound_g = 1 / math.sqrt(fan_in_g)
            bound_b = 1 / math.sqrt(fan_in_b)
            init.uniform_(self.bias_r, -bound_r, bound_r)
            init.uniform_(self.bias_g, -bound_g, bound_g)
            init.uniform_(self.bias_b, -bound_b, bound_b)
            init.uniform_(self.bias_b, -bound_g, bound_g)

    def forward(self, input):
        y1 = F.linear(input, self.weight_r, self.bias_r)
        y2 = F.linear(input, self.weight_g, self.bias_g)
        y3 = F.linear(input**2, self.weight_b, self.bias_b)
        return y1 * y2 + y3
        #return y1 * y2 + self.bias_b

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    

##################################    

class General_quadratic(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=False):
        super(General_quadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_a = Parameter(torch.Tensor(out_features, in_features, in_features))
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / self.weight_a.size(1)
        init.normal_(self.weight_a, std = bound)
        init.normal_(self.weight, -math.sqrt(bound), math.sqrt(bound))
        #init.uniform_(self.weight_a, -bound, bound)
        for i in range(self.out_features):
            self.weight_a.data[i,:,:] = (self.weight_a.data[i,:,:]+self.weight_a.data[i,:,:].T)/2
            
        if self.bias is not None:
            init.uniform_(self.bias, -math.sqrt(bound), math.sqrt(bound))

    def forward(self, input):
        y = F.bilinear(input, input, self.weight_a)
        y1 = F.linear(input, self.weight, self.bias)
        #return y
        return (y+y1)/2

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    
############################################    
    
    
class Low_dimensional_quadratic(Module):
   
    __constants__ = ['bias', 'in_features', 'out_features', 'number_eigens']

    def __init__(self, in_features, out_features, number_eigens, bias=False):
        super(Low_dimensional_quadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.number_eigens = number_eigens
        self.eigen = Parameter(torch.Tensor(out_features, in_features, number_eigens))
        self.bias = Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.eigen.size(1))
        init.kaiming_normal_(self.eigen, a=10)
        #init.normal_(self.eigen, std = bound)    

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.eigen)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        if self.number_eigens == 1:
            x = F.linear(input, self.eigen[:,:,0])*F.linear(input, self.eigen[:,:,0])
        else:
            x = F.linear(input, self.eigen[:,:,0])-F.linear(input, self.eigen[:,:,0])
            for i in range(self.number_eigens):
                x += F.linear(input, self.eigen[:,:,i])*F.linear(input, self.eigen[:,:,i])
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, number_eigens={}, bias={}'.format(
            self.in_features, self.out_features, self.number_eigens, self.bias is not None
        )


############################################   


class Dales_General_quadratic(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, EI_distribution, bias=False):
        super(Dales_General_quadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.EI_distribution = EI_distribution
        self.weight_a = Parameter(torch.Tensor(out_features, in_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight_a.size(1))
        kappa_matrix = -torch.ones(self.in_features, self.in_features)
        kappa_matrix[self.EI_distribution==0,:] = 1
        kappa_matrix[:,self.EI_distribution==0] = 1
        
        init.uniform_(self.weight_a, -bound, bound)
        #init.uniform_(self.weight, -bound, bound)
        for i in range(self.out_features):
            self.weight_a.data[i,:,:] = abs(self.weight_a.data[i,:,:]+self.weight_a.data[i,:,:].T)*kappa_matrix/2


    def forward(self, input):
        y = F.bilinear(input, input, self.weight_a)
        #y1 = F.linear(input, self.weight, self.bias)
        return y
        #return y+y1

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    
############################################    
    