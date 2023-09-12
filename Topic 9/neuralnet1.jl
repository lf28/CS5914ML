### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ab7b3ac3-953d-41fb-af70-7e92ab06b8c7
begin
	using Zygote
	using Flux
	using Flux: logitcrossentropy, normalise, onecold, onehotbatch
	using DataFrames
	using Random
	using StatsPlots
	using Latexify, LaTeXStrings
	using Statistics: mean
	using PlutoTeachingTools
	using PlutoUI
	using LinearAlgebra
	using LogExpFunctions
	# using StatsBase
	using Distributions
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style	
end

# ╔═╡ 28754109-9d20-4c20-a2b1-491dd56dcc2f
using Images

# ╔═╡ f72cacd8-f5c2-43a7-b737-71c6b5185880
begin
	using Logging
	Logging.disable_logging(Logging.Info); # or e.g. Logging.Info
end;

# ╔═╡ 50103dcc-034e-4c22-953c-1b5fd781c070
TableOfContents()

# ╔═╡ fdab762e-4d32-43f9-8a0d-267fec8e491e
ChooseDisplayMode()

# ╔═╡ 78aad856-dc62-430d-ad3a-a32b0d268b0d
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ a9c35a78-c179-45f4-ba25-18e744b30129
md"""

# CS5914 Machine Learning Algorithms


#### Neural Network

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 1e180936-7d39-4895-b526-9a09d7024946
md"""

## This time

An introduction to Neural Networks (NN)
* intuitively: stack multiple single neurons together 


* some commonly used NN constructs and their implementation
  * Linear layer 
  * activation functions


* and their implementations from scratch


* regularisation techniques of neural networks
\

## Next time

Other topics to discuss (Next topic)


* learning: backpropagation (next time)
* some other tricks and practices: vanishing gradient, batch normalisation
* advanced gradient descent algorithms
"""

# ╔═╡ 16a460dd-a1bb-4304-bd80-cdeaa82c1cac
md"""

## Recap: logistic regression


Logistic regression is also known as a single-neuron 

```math
\large
\textcolor{darkorange}{\sigma(\mathbf{x})} = \textcolor{darkorange}{\texttt{logistic}}\left (\underbrace{\color{darkblue}{\begin{bmatrix} w_1&  w_2& \ldots&   w_m \end{bmatrix}}  \color{darkgreen}{\begin{bmatrix} x_1\\  x_2\\ \vdots\\   x_m \end{bmatrix}} + \textcolor{darkblue}{b} \color{black}}_{z} \right )
```
* ``\color{Periwinkle}z = {\color{darkblue}\mathbf{w}}^\top{\color{darkgreen}\mathbf{x}} +{\color{darkblue}b}`` : pre-activation (a hyperplane)
* an activation function: ``a``: ``\sigma(\cdot) \in (0, 1)``:  ${\texttt{logistic}}(z) =\frac{1}{1+e^{-z}}$
"""

# ╔═╡ 70d6cadd-04e4-4bb5-b38e-8d510a3f2bcf
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/single_logreg.png
' width = '500' /></center>"

# ╔═╡ f413ae69-9df1-4447-84b0-8f35b39f55cd
md"""
## Recap: logistic regression


"""

# ╔═╡ 5262a879-a236-44ee-8d00-ef8547de579b
wv_ = [1, 1] * 1

# ╔═╡ 78d6618f-cd68-434b-93e4-afd36c9e3247
md"""

## Recap: single neuron (linear regression)



A **linear regression** can be viewed as a _special case_, where

```math
\large
\sigma(\mathbf{x}) = \texttt{identity}(\mathbf{w}^\top\mathbf{x})
```
* activation function is a boring _identity_ function: $\texttt{identity}(z) =z$


"""

# ╔═╡ 0d0b78f8-8b53-4f11-94d9-d436c84ae976
md"""

```math
\textcolor{darkorange}{\sigma(\mathbf{x})} = \texttt{I}\left (\color{darkblue}{\begin{bmatrix} w_1&  w_2& \ldots&   w_m \end{bmatrix}}  \color{darkgreen}{\begin{bmatrix} x_1\\  x_2\\ \vdots\\   x_m \end{bmatrix}} + {\color{darkblue} b} \color{black} \right );
```

"""

# ╔═╡ a7b797a8-d791-4d10-9ac2-7bd9c0815688
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/single_linreg.png
' width = '400' /></center>"

# ╔═╡ bbb08fad-6d96-4735-9ac5-a03f27773af1
md"""

## Recap: binary classification cross entropy loss


Binary classification usually use binary cross entropy (BCE) loss:


```math
\large
\begin{align}
 \text{BCE\_loss}(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{i=1}^n {y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})
\end{align}
```
"""

# ╔═╡ 7159fa02-e801-4a86-8653-aae1dda2e7ac
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/logistic_reg_loss.png
' width = '600' /></center>"

# ╔═╡ f697e1bb-4e8b-458c-8ea1-d85c403fd913
md"""

## Recap: binary classification learning


"""

# ╔═╡ 0b945b04-2808-4e81-9c2f-18ef5c891366
md"""

Based on the BCE loss, we have derived the gradients for ``\{\mathbf{w}, b\}`` 


```math
\large
\begin{align}
\nabla_{{\mathbf{w}}}L(y, \hat{y}) &= - (y- \hat{y})\cdot \mathbf{x} \\
\nabla_{b}L(y, \hat{y}) &= - (y- \hat{y})\cdot 1

\end{align}
```

* where $\hat{y} = \sigma(\mathbf{w}^\top\mathbf{x}+b)$ is the prediction output


##### Show me the code

_Gradient descent_

```python
while True:
	gradw, gradb = evaluate_gradient(train_x, train_y, w, b)
	w -= lr * gradw
	b -= lr * gradb
```
"""

# ╔═╡ f510ba7f-8ca5-42ed-807d-67a8710d2e00
aside(tip(md"""
If we use the dummy predictor notation, the gradient simplified to


```math
\nabla_{\tilde{\mathbf{w}}}L(y, \hat{y}) = - (y- \hat{y})\cdot \mathbf{x},
```

where ``\mathbf{x} = [1, \mathbf{x}]^\top`` and ``\tilde{\mathbf{w}} \triangleq [b, \mathbf{w}]^\top``

"""))

# ╔═╡ 45788db8-4d37-4b7a-aadb-6c6993bf37e8
md"""

## Recap: binary classification learning demonstration


"""

# ╔═╡ f000ec11-e922-4262-b985-3882b2e6c2ee
md"""

## However 



#### Real world data sets are rarely _linear_

* ###### a single neuron model is rarely enough
"""

# ╔═╡ c95275ed-b682-49b8-ae54-9e9e84d401e6
md"""
##

##### Non-linear regression 
\

A non-linear regression example
"""

# ╔═╡ 7dcff9b2-763b-4424-97d7-48da1e3d2aff
begin
	function true_f(x)
		-5*tanh(0.5*x) * (1- tanh(0.5*x)^2)
	end
	Random.seed!(100)
	x_input = collect(range(-8, 8, 50))
	
	y_output = true_f.(x_input) .+ sqrt(0.05) * randn(length(x_input))
	# xμ, xσ = mean(x_input), sqrt(var(x_input))
	# x_input = (x_input .- xμ) / xσ
end;

# ╔═╡ 1d968144-0bf7-4e08-9600-6fc33cdbdb52
begin
	gr()
	scatter(x_input, y_output, label="Observations", title="A non-linear regression dataset")	
	plot!(true_f, lw=2, xlim=[-10, 10], framestyle=:default,  lc=:gray, ls=:dash,label="true function", xlabel=L"x", ylabel=L"y")
end

# ╔═╡ 6d84e048-d215-4597-b1f9-dbc24e031791
md"""

##

##### Non-linear classification

\

A non-linear decision boundary example

"""

# ╔═╡ 2945e298-1983-4e6b-a903-bead1ac833a8
begin
	#Auxiliary functions for generating our data
	function generate_real_data(n)
	    x1 = rand(1,n) .- 0.5
	    x2 = (x1 .* x1)*3 .+ randn(1,n)*0.1
	    return vcat(x1,x2)
	end
	function generate_fake_data(n)
	    θ  = 2*π*rand(1,n)
	    r  = rand(1,n)/3
	    x1 = @. r*cos(θ)
	    x2 = @. r*sin(θ)+0.5
	    return vcat(x1,x2)
	end
	gr()
	# Creating our data
	train_size = 1000
	real = generate_real_data(train_size)
	fake = generate_fake_data(train_size)
	test_size = 1000
	real_test = generate_real_data(test_size)
	fake_test = generate_fake_data(test_size)
	# Visualizing
	scatter(real[1,1:500],real[2,1:500], label="class 1", title="A non-linear classification")
	scatter!(fake[1,1:500],fake[2,1:500], ratio=1, label="class 2", framestyle=:origin)
end

# ╔═╡ 1fb81ca3-2fa9-4862-bad1-00446515a23a
md"""

## Another non-linear example


"""

# ╔═╡ 35054345-67ff-4a5f-8b01-7f6ddf087fcc
md"""
##
"""

# ╔═╡ fe45432f-9224-4e1e-a981-0136db240085
md"""

## How about two neurons?


Each pair of datasets can be handled by one neuron
* *i.e.* linearly separable

* just stack two neurons together!
"""

# ╔═╡ 6886fe8d-9645-4094-85f8-6b3fb5d409e4
md"""

## How about two neurons? (conti.)

##### ``z_1, z_2`` are two neuron's hyperplanes (pre-activations)
\

```math 
\Large
\begin{align}
z_1 = \mathbf{w}_1^\top\mathbf{x} +b_1\\ 
z_2 = \mathbf{w}_2^\top \mathbf{x} + b_2
\end{align}
``` 

"""

# ╔═╡ f52e374a-2031-4bd9-b58c-16598ffac15a
TwoColumn(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-01.png", :height=>300), Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-02.png", :height=>300))

# ╔═╡ 4b2fd42a-e88d-4117-af79-84cfaf902122
md"""

## Matrix notation
"""

# ╔═╡ 5bdd174d-9250-4864-80af-1360240ccc93

md"""
``z_1, z_2`` are two neuron's hyperplanes

```math 
\large
\begin{align}
z_1 = \mathbf{w}_1^\top\mathbf{x} +b_1\\ 
z_2 = \mathbf{w}_2^\top \mathbf{x} + b_2
\end{align}
``` 

which can compactly written in matrix notation


```math
\large
\underbrace{\begin{bmatrix}
z_1 \\
z_2
\end{bmatrix}}_{\mathbf{z}} = \underbrace{\begin{bmatrix} \rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_1^\top & \rule[.5ex]{2.5ex}{0.5pt} \\

\rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_2^\top & \rule[.5ex]{2.5ex}{0.5pt}
\end{bmatrix}}_{\mathbf{W}} \underbrace{\begin{bmatrix} x_1 \\

x_2
\end{bmatrix}}_{\mathbf{x}} + \underbrace{\begin{bmatrix} b_1 \\
b_2
\end{bmatrix} }_{\mathbf{b}}
```

"""

# ╔═╡ 36a00007-0da1-48a4-a225-d400d9c61a37
md"""

## Matrix notation
"""

# ╔═╡ ca75fcaf-9b6d-4d1d-a97e-1b0167b1e2d8

md"""
``z_1, z_2`` are two neuron's hyperplanes

```math 
\large
\begin{align}
z_1 = \mathbf{w}_1^\top\mathbf{x} +b_1\\ 
z_2 = \mathbf{w}_2^\top \mathbf{x} + b_2
\end{align}
``` 

which can compactly written in matrix notation


```math
\large
\underbrace{\begin{bmatrix}
z_1 \\
z_2
\end{bmatrix}}_{\mathbf{z}} = \underbrace{\begin{bmatrix} \rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_1^\top & \rule[.5ex]{2.5ex}{0.5pt} \\

\rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_2^\top & \rule[.5ex]{2.5ex}{0.5pt}
\end{bmatrix}}_{\mathbf{W}} \underbrace{\begin{bmatrix} x_1 \\

x_2
\end{bmatrix}}_{\mathbf{x}} + \underbrace{\begin{bmatrix} b_1 \\
b_2
\end{bmatrix} }_{\mathbf{b}}
```

or even better

```math
\LARGE
\texttt{Linear}(\mathbf{x}, \mathbf{W}, \mathbf{b}) =\boxed{\mathbf{z} = \mathbf{W} \mathbf{x} +\mathbf{b}}
```

* we name the function _Linear Layer_
* where note that

```math

\mathbf{W} = \begin{bmatrix} w_{11} & w_{12} \\
w_{21} & w_{22}
\end{bmatrix} =\begin{bmatrix} \rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_1^\top & \rule[.5ex]{2.5ex}{0.5pt} \\

\rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_2^\top & \rule[.5ex]{2.5ex}{0.5pt}
\end{bmatrix}
```

* each _row_ is a neuron's weight parameter vector
* we have two neurons: two rows in ``\mathbf{W}``

"""

# ╔═╡ f6e77a11-8e1a-44e2-a2d7-cd29939dcc30
md"""

## Linear layer (generalisation)


##### ``n`` neuron Linear Layer generalisation

* input: ``\mathbf{x} \in \mathbb{R}^m``
* output: ``\mathbf{z} =[z_1, z_2, \ldots, z_n] \in \mathbb{R}^n``
"""

# ╔═╡ 001a7866-f0c4-4cea-bda3-d5fea2ce56b4
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear1.svg
' height = '400' /></center>"

# ╔═╡ 2271ac6d-4bb7-4683-8672-c280c3e57fd3
md"""

## Linear layer (generalisation)


##### ``n`` neuron Linear Layer generalisation

* input: ``\mathbf{x} \in \mathbb{R}^m``
* output: ``\mathbf{z} =[z_1, z_2, \ldots, z_n] \in \mathbb{R}^n``
"""

# ╔═╡ 350146ba-6e11-41f8-b451-b3c842722de9
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear2.svg
' height = '400' /></center>"

# ╔═╡ fcb991b7-e3e1-4f82-8a8b-c1f2a99d6cce
md"""

## Linear layer (generalisation)


##### ``n`` neuron Linear Layer generalisation

* input: ``\mathbf{x} \in \mathbb{R}^m``
* output: ``\mathbf{z} =[z_1, z_2, \ldots, z_n] \in \mathbb{R}^n``
"""

# ╔═╡ 8971420b-7687-4a89-8cfa-da2483be8730
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear3.svg
' height = '400' /></center>"

# ╔═╡ a7051da0-f28a-4261-b71c-f7450341e1bf
# md"""

# ## Linear Layer implementation


# Just a many to many function


# ```math
# \LARGE
# \texttt{Linear}(\mathbf{x}; \mathbf{W}, \mathbf{b}) = \mathbf{W} \mathbf{x} +\mathbf{b}
# ```


# * ``\mathbf{W}``: size ``n \times m``
#   * ``n``: # of hidden neurons
#   * ``m``: # of input (or input dimension)


# * ``\mathbf{b}``: size ``n\times 1``
#   * bias vector: one bias for each of the ``n`` neurons

# """

# ╔═╡ 500d6f5f-30ee-45a0-9e3c-5f8ba5274e75
begin
	# That's it!
	function denselayer(X, W, b; σ = identity)
		h = W * X .+ b
		σ.(h)
	end
end;

# ╔═╡ 2ec97a4c-0ddc-4918-91a1-0bf177b4a202
md"""

## Linear Layer implementation 


In maths, Linear layer is 


```math
\LARGE
\texttt{Linear}(\mathbf{x}; \mathbf{W}, \mathbf{b}) = \mathbf{W} \mathbf{x} +\mathbf{b}
```


* ``\mathbf{W}``: size ``n \times m``
  * ``n``: # of output hidden neurons
  * ``m``: # of input (or input dimension)


* ``\mathbf{b}``: size ``n\times 1``
  * bias vector: one bias for each of the ``n`` neurons


## Linear Layer implementation in Python


However, **Python** (PyTorch, numpy) is a **row-major** programming language

* a vector ``\texttt{x}`` is assumed a row vector by default


Therefore, when it comes to implementation in Python

```math
\LARGE
\texttt{z}_{1\times n} = \texttt{x}_{1\times m} @ \texttt{W}_{m\times n}  +\texttt{b}_{1\times n}
```


* that is ``\texttt{z} = \mathbf{z}^\top``, ``\texttt{W} = \mathbf{W}^\top``, ``\texttt{b} =\mathbf{b}^\top``
"""

# ╔═╡ b7d84cb8-59ee-41e6-a04e-a4be9f905f95
md"""

In particular, $\texttt{W}_{\text{in} \times \text{out}} \in \mathbb{R}^{\text{in} \times \text{out}}$

$$\texttt{W}= \begin{bmatrix} \mid &\mid &  \mid &\mid \\
\mathbf{w}_1 &\mathbf{w}_2 & \ldots & \mathbf{w}_{\text{out}} \\
\mid &\mid &  \mid &\mid 
\end{bmatrix}$$
"""

# ╔═╡ d6023ad2-2467-44a4-8c57-d12024e76536
aside(tip(md"""
```math
\begin{align}
\mathbf{z} &=  \mathbf{Wx} + \mathbf{b}\\
\mathbf{z}^\top &= \mathbf{x}^\top \mathbf{W}^\top + \mathbf{b}^\top
\end{align}
```
"""))

# ╔═╡ 82a2a13c-cdbe-4901-8b2e-86a5d3cd99e5
md"""

!!! note "Linear Layer" 
	```python
	class Linear:  
	    def __init__(self, in_size, out_size):
	        self.weight = torch.randn((in_size, out_size), generator=g) 
	        self.bias = torch.zeros(out_size) 
	  
	    def __call__(self, x):
	        self.out = x @ self.weight + self.bias
	        return self.out

	    ...
	```

"""

# ╔═╡ afce64cb-cdca-42a1-8e92-f1e9634af976
md"""


```python
l1 = Linear(2, 2) # create a 2 to 2 linear layer
z = l1(torch.rand(2))
```
"""

# ╔═╡ 7841a726-072a-41a3-a10e-4a4a5256aef1
md"""
## Linear Layer implementation in Julia*

Julia is a modern language designed for numerical computing

* the implementation is just direct translation from maths
"""

# ╔═╡ ddb1bd96-e316-450a-a7b5-f5561a890266
begin
	struct Linear
		W
		b
	end

	# constructor with input and output size and identity act
	Linear(in::Integer, out::Integer) = Linear(randn(out, in), randn(out))
	
	# Overload function, so the object can be used as a function
	(m::Linear)(x) = m.W * x .+ m.b
end;

# ╔═╡ 0fe62e04-559d-4140-970b-6baca845be63
Linear(2, 2)(rand(2));

# ╔═╡ 8d009757-ab79-47e5-8be0-080ef091b302
begin
	# That's it!
	function linearlayer(X, W, b)
		z = W * X .+ b
	end
end;

# ╔═╡ f079377b-d71f-4f70-aece-f04d12cd3127
md"""

## Activation functions

We also apply some non-linear activation functions **element-wisely** to ``\mathbf{z}``


"""

# ╔═╡ fe63494b-716c-4718-afcd-026291dc7b94
TwoColumn(md"""
\
\
\

```math 
\large
\begin{align}
a(z_1) = \sigma(\underbrace{\mathbf{w}_1^\top\mathbf{x} +b_1}_{z_1})\\ 
a(z_2) = \sigma(\underbrace{\mathbf{w}_2^\top \mathbf{x} + b_2}_{z_2})
\end{align}
``` 

""", 

Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear5_act.svg", :height=>300)
)

# ╔═╡ e6d3f1d1-21cb-4662-8806-77b88038be7c
md"""

which can be written as 


```math
\large
\mathbf{a} = \sigma.(\mathbf{Wx}+\mathbf{b})
```

* ``\sigma\,`` +``\,`` dot : element-wise operation, apply the function to each element
* some use ``\odot`` to denote element-wise operation, *e.g.* ``\mathbf{a} = \sigma \odot (\mathbf{Wx}+\mathbf{b})``
"""

# ╔═╡ cb1bc71b-8c8b-4873-8f28-2935c0972b8c
md"""

## Activations

##### It is also common to draw the activation and linear layer together 
"""

# ╔═╡ 4a0195bf-2d5c-478b-b571-edd4b3e532eb
TwoColumn(html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear5_act.svg
' height = '300' /></center>" , html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-03.png
' height = '255' /></center>")

# ╔═╡ 13786181-9280-4260-9705-f2aa419e9c26
md"""
## Linear layer + activations


Now consider ``m``-dimensional input and ``n`` neurons

* ``m`` input/predictors
* ``n`` output hidden neurons


##### We use _super-index_ ``^{(l)}`` to index the layer of a NN
"""

# ╔═╡ fec26acd-d545-428d-a90d-17a89b2e11c9
ThreeColumn(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-08.png", :height=>350), Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-09.png", :height=>350), Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-10.png", :height=>350))

# ╔═╡ 6d443b0c-becc-459a-b4ee-9a863c7d69b3
md"""

## Linear Layer + activation


**_Linear_** Layer + element-wise **activation** is a ``\mathbb{R}^m \rightarrow \mathbb{R}^n`` function


```math
 \mathbf{x} \mapsto \sigma.(\texttt{Linear}(\mathbf{x}; \mathbf{W}, \mathbf{b}))
```

"""

# ╔═╡ be7087be-f948-4701-b5c2-00580ca27d9b
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-11.png
' height = '400' /></center>"

# ╔═╡ c98638f4-5bb1-4550-aa51-945c4632b801
md"""

## Activation functions


Some common activation functions are
* `relu` is always a good and safe choice if not sure



"""

# ╔═╡ eef80655-c86f-4947-be7e-ec5b7ff13ad2
ThreeColumn(
md"""


**Identity**: 

```math
\small
\sigma(z) = z
```

**Logistic**: 

```math
\small

\sigma(z) = \frac{1}{1+e^{-z}}
```

**Tanh**: 

```math
\small

\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z}+ e^{-z}}
```



""",


md"""


**ReLu** (Rectified linear unit): 

```math
\small
\begin{align}
\text{ReLu}(z) &= \begin{cases}0 & z < 0\\ z & z\geq 0 \end{cases}\\
&=\max(z, 0)

\end{align}
```


**Leaky ReLu**: 

```math
\scriptsize
\text{LeakyReLu}(z) = \begin{cases}0.01 z & z < 0\\ z & z\geq 0 \end{cases}
```


**softplus**: 

```math
\small

\text{softplus}(z) = \ln(1+e^z)
```

"""
	
	
	
	,
	
md"""


**Gelu** (Gaussian Error Linear Unit): 

```math
\scriptsize
\begin{align}
\text{Gelu}(z) &= 0.5x *\\
&\;\;(1 + \tanh(\sqrt{\tfrac{2}{\pi}} (x + 0.044715x^3)))
\end{align}
```


**Swish**: 

```math
\small

\text{swish}(z) =  \frac{z}{1+e^{-z}}
```


**CeLu**: 

```math
\small

\text{Celu}(z, \alpha) =  \begin{cases}z & z \geq 0\\ \alpha\left(e^{z/\alpha} -1\right) & \text{otherwise}\end{cases}
```

"""

	
)

# ╔═╡ 8aa1f78a-9cf8-4186-b9a0-31b8a00eabfc
let
	gr()
	p_relu = plot(relu, lw=1.5, lc=1, label=L"\texttt{ReLu}(z)",legend=:topleft)
	p_logis = plot(logistic, lw=1.5, lc=3,label=L"\texttt{logistic}(z)", legend=:topleft)
	plot!(tanh, lw=1.5, lc=2, label=L"\tanh(z)",legend=:topleft)
	# plot!(p_relu, leakyrelu, lw=1.5, lc=4, label=L"\texttt{LReLu}(z)", legend=:topleft)
	plot!(p_relu, Flux.softplus, lw=1.5, lc=4, label=L"\texttt{softplus}(z)", legend=:topleft)
	p_gelu = plot(gelu, lw=1.5, lc=6, label=L"\texttt{gelu}(z)", legend=:topleft)
	plot!(swish, lc=7, label=L"\texttt{swish}(z)", lw=2)

	p_lrelu = plot(leakyrelu, lw=1.5, lc=6,label=L"\texttt{LeakyReLu}(z)", legend=:topleft)

	plot!(celu, lc=7, lw=1.5,label=L"\texttt{celu}(z)")
	plot(p_logis, p_relu, p_gelu, p_lrelu)
end

# ╔═╡ 81c2fa7b-d3b3-4fe3-b2e6-831a10616ed0
# md"
# A more user-friendly implementation

# * use `struct`
# * a layer with ``n`` neurons can be constructed specifically
# * the function map is overloaded
# "

# ╔═╡ 361d76ed-7952-4c64-830c-27c6273272d7
md"""

## Implementations -- Python
"""

# ╔═╡ 7001dfa7-fb39-40fc-acac-750c7f0d6605
md"""

As an example, ReLu can be implemented as

```python
class ReLu:
    def __call__(self, x):
        self.out = torch.clip(x, 0) ## clip will return the max between x and 0
        return self.out
```

or simply use the built-in method

```python
class ReLu:
    def __call__(self, x):
        self.out = torch.relu(x) ## clip will return the max between x and 0
        return self.out
```
"""

# ╔═╡ 743030fd-4462-4563-8767-2d4bfbb17f5b
md"""

```python
l1 = Linear(2, 5) # a linear layer with 2 input and 5 neurons output
relu = ReLu() # instantiate a ReLu() activation function
l1relux = relu((l1(x))) # element wise activations
```
"""

# ╔═╡ f3ca219a-e2af-45f7-85bf-3c050227a33b
md"""

## Implementation -- Julia*

Julia's implementation simply merge _Linear_ layer and activation together
* it is easier to derive the gradient 
"""

# ╔═╡ 19142836-f9c0-45c1-a05b-24d810a06f8f
begin
	struct DenseLayer
		W
		b
		act::Function
	end

	# constructor with input and output size and identity act
	DenseLayer(in::Integer, out::Integer) = DenseLayer(randn(out, in), randn(out), identity)

	# constructor with non-default act function
	DenseLayer(in::Integer, out::Integer, a::Function) = DenseLayer(randn(out, in), randn(out), a)
	
	# Overload function, so the object can be used as a function
	(m::DenseLayer)(x) = m.act.(m.W * x .+ m.b)
end;

# ╔═╡ a5efe47c-71fb-4b1e-8417-47036b96238d
md"""
## Finalising: add output layer (regression)

For **regression**, we just output the final output as it is

* just another **linear layer** *without* activation

"""

# ╔═╡ bd6ccedb-00a7-4f9d-8ec9-861d8a8ada11
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg.svg
' height = '345' /></center>"

# ╔═╡ 378c520d-881c-4056-a3d0-03fcddf2126c
md"""

## Why (non-linear) activations ?



"""

# ╔═╡ dbddfc50-5d66-45c9-b61a-a554282e38c7
TwoColumn(md"""
###### _Non-linear_ activations are essential for neural networks

* it makes the final function **non-linear**


###### **Without** _non-linear_ *activations*, a neural network becomes another linear model

\

To see this, we set ``\sigma = \texttt{idensity}``, *i.e.* no non-linear activations, then

```math
\mathbf{a} = \mathbf{Wx}+\mathbf{b} 
```



""", html"<br><br><br><br><br><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg.svg
' width = '345' /></center>")

# ╔═╡ 0f6a50ea-6165-4e9b-b332-641fed50eea9
md"""

## Why (non-linear) activations ?


"""

# ╔═╡ bad0d1a7-f93a-4af8-a453-4ea8e1385d7c
TwoColumn(md"""
###### _Non-linear_ activations are essential for neural networks

* it makes the final function **non-linear**


###### **Without** _non-linear_ *activations*, a neural network becomes another linear model

\

To see this, we set ``\sigma = \texttt{idensity}``, then

```math
\mathbf{a} = \mathbf{Wx}+\mathbf{b} 
```

and

```math
\begin{align}
z &= \mathbf{v}^\top (\underbrace{ \mathbf{Wx}+\mathbf{b}}_{\mathbf{a}})+ b_0 \\
\end{align}
```

""", html"<br><br><br><br><br><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg.svg
' width = '345' /></center>")

# ╔═╡ c6de70e6-14f4-4b1a-8097-4e086a7fedde
md"""

## Why (non-linear) activations ?


"""

# ╔═╡ bb317788-8d04-45f8-84f7-7a0226a7df6f
TwoColumn(md"""
###### _Non-linear_ activations are essential for neural networks

* it makes the final function **non-linear**


###### **Without** _non-linear_ *activations*, a neural network becomes another linear model

\

To see this, we set ``\sigma = \texttt{idensity}``, then

```math
\mathbf{a} = \mathbf{Wx}+\mathbf{b} 
```

and

```math
\begin{align}
z &= \mathbf{v}^\top (\underbrace{ \mathbf{Wx}+\mathbf{b}}_{\mathbf{a}})+ b_0 \\
&= \underbrace{\mathbf{v}^\top \mathbf{W}}_{\boldsymbol{\omega^\top}}\mathbf{x} + \underbrace{\mathbf{v}^\top \mathbf{b} +b_0}_{\omega_0} \\
&= \boldsymbol{\omega}^\top\mathbf{x}+ \omega_0 \;\; \text{\# a linear function}
\end{align}
```

""", html"<br><br><br><br><br><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg.svg
' width = '345' /></center>")

# ╔═╡ 939c8687-3620-4e98-b9f5-0c304312ef2f
md"""

## Finalising: add output layer (classification)

For binary **classification**, the output should be a scalar (between 0 and 1)

* the same idea as logistic regression: apply logistic transformation


"""

# ╔═╡ a617405e-fc19-4032-8642-7f01fb486c74
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-05.png
' height = '350' /></center>"

# ╔═╡ a8f32a69-3150-4639-9cd4-546ab8442d90
md"""

## An example

##### A showllow neural network with three hidden units
"""

# ╔═╡ 6daa681e-3974-497c-baec-347740b2a29e
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowNet.svg
' height = '350' /></center>"

# ╔═╡ 3db05083-dc9e-49d9-9d11-68562eac5827
md"""

##### The pre-activation functions
"""

# ╔═╡ 8674f942-dfae-4891-b3e1-a38b79a2b621
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowBuildUp0.svg
' width = '750' /></center>"

# ╔═╡ 1be45d72-0661-4a3a-875f-888d0c12b5e4
md"""
##
##### The after-activation hidden layers (ReLu)
"""

# ╔═╡ d6353c42-b07b-483a-a669-857c480385b9
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowBuildUp1.svg
' width = '750' /></center>"

# ╔═╡ 0e929a8d-2c43-4e51-b990-8855963b4b4d
md"""
##
##### The final output
"""

# ╔═╡ 0fe8412f-4dc8-4698-b434-81763038e768
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowBuildUp1.svg
' width = '750' /></center>"

# ╔═╡ 6a4faffd-39e9-4576-a820-3f3abb724055
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowBuildUp2.svg
' width = '750' /></center>"

# ╔═╡ a9427f86-20d2-43ca-b1ca-9724252315da
md"""

[Source: Understanding Deep Learning](https://udlbook.github.io/udlbook/)
"""

# ╔═╡ afe83e7a-5e5f-4fe4-aa82-082c7fcdfed3
md"""

## Universal approximator



##### Neural network is *universal function approximator*
- A neural network with *infinite* number of neurons can approximate any function arbitrarily well




"""

# ╔═╡ 13e0cd61-2f3f-40be-ad4a-7a2fb13661bf
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowApproximate.svg
' width = '750' /></center>"

# ╔═╡ a594cfb0-bd4d-427e-ba00-c665a2c41f54
md"""

## Fixed basis regression vs Neural network



"""

# ╔═╡ 9a97cc7a-3b3f-41bf-a41a-80dde51a01cc
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/fixedbasisreg_.svg
' height = '450' /></center>"

# ╔═╡ 9907c198-8f88-46cf-804d-ba6bde31770b
md"""

Fixed basis regression is a specific case (more restricted version) of Neural network

* one hidden layer

* and the hidden layers' parameters are fixed

$\{\mathbf{W}, \mathbf{b}\} \text{: are fixed rather than adaptively learnt}$

"""

# ╔═╡ 61873a36-2262-4a3b-bb2d-83417c76914a
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/fixed_basis_mlp_.svg
' width = '450' /></center>"

# ╔═╡ 3fde5353-863b-4fd0-8ce2-c6c26d934a14
md"""

# Deep neural network
"""

# ╔═╡ cf69619e-9313-4d14-826b-9a249e1e6b06
md"""

## Make it deep



"""

# ╔═╡ 2c42c70f-05d9-4d6e-8ddd-d225d65a39a6
TwoColumn(md"""

\
\
\
\


We can add more than one hidden layer


* hidden layer ``(l)`` to layer ``(l+1)``

  * inputs: ``\mathbf{a}^{(i)} \in \mathbb{R}^m``
  * output: ``\mathbf{a}^{(i+1)} \in \mathbb{R}^n``

* super-index ``(l)`` index the i-th layer of a NN

""", 
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-13.png
' height = '450' /></center>"
)

# ╔═╡ e1a62d74-aaae-491e-b5ef-89e3c1516ca9
md"""

## Make it deeper


Here, we introduce an **additional hidden layer**

* ##### this is a neural network with two hidden layers
* also known as **Multi-layer perceptron (MLP)**
"""

# ╔═╡ 8b66daf7-4bf7-4323-be48-a6defefcf022
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp20.svg
' width = '450' /></center>"

# ╔═╡ 91deb3a5-5a59-4a71-a5f0-8037024e1e97
md"""

## Make it deeper


Here, we introduce two **additional hidden layer**

* ##### this is a neural network with three hidden layers
"""

# ╔═╡ 1fe38da7-09ca-4006-8c18-97f4a5c5ce78
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp21.svg
' width = '500' /></center>"

# ╔═╡ 9dea643a-9f72-404b-b57f-496933cc8013

md"""
## Why deeper is better ?


#### A _deep-ish_ network
* with a bottlenet middle layer with one neuron (with identity activation)
* **two** neural networks 
"""

# ╔═╡ 73b965c6-4f67-4130-ba81-e72b809b00b6
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/DeepConcat0.svg
' width = '700' /></center>"

# ╔═╡ b5a49d08-1786-405b-8b6c-17a9ead69cc2
md"""
## Why deeper is better ?


#### A _deep-ish_ network
* with a bottlenet middle layer with one neuron (with identity activation)
* **two** neural networks 
"""

# ╔═╡ 5e7d2395-0095-4c3f-bf21-b2c99fa34e4f
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/DeepConcat1.svg
' width = '700' /></center>"

# ╔═╡ f3f3036f-2744-47bc-95a6-6512460023f7
md"""
## Why deeper is better ?


#### A _deep-ish_ network
* with a bottlenet middle layer with one neuron (with **identity activation**)
* **two** neural networks **composed together**
"""

# ╔═╡ f815155f-7d28-473d-8fa9-e0e7f7d81df3
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/DeepConcat2.svg
' width = '700' /></center>"

# ╔═╡ ca91ae11-7b2f-4b83-a06c-e66a51ec53a7
md"""
## Why deeper is better ?


#### A _deep-ish_ network
* with a bottlenet middle layer with one neuron (with **identity activation**)
* the final function when **two** neural networks **composed together**
  * #####   ``3^2=9`` regions!
  * comparing with a shallow net with ``6`` neurons

"""

# ╔═╡ 5159ff24-8746-497b-be07-727f9b7a4d82
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/DeepConcat.svg
' width = '700' /></center>"

# ╔═╡ 6a20011e-c89b-46b4-a1f9-243bd401c8fa
md"""

[Source: Understanding Deep Learning](https://udlbook.github.io/udlbook/)
"""

# ╔═╡ 0fbac3bc-3397-49f6-9e9d-1e31908f530e
md"""

## Neural Networks implementation


"""

# ╔═╡ 08ed93ed-304a-46a5-bfc3-c8f45df246ed
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp21.svg
' height = '220' /></center>"

# ╔═╡ 96e3e41c-0c50-4e9b-b308-2b307fe54fc8
TwoColumn(md"""

### Show me the maths

Neural network is a ``\mathbb{R}^m \rightarrow \mathbb{R}^n`` function 




```math
\begin{align}
\texttt{nnet}(\mathbf{x}) &= \texttt{Layer}^{(L)}(\\
&\;\;\;\;\;\texttt{Layer}^{(L-1)}(\\
&\;\;\;\;\;\;\;\;\; \vdots \\
&\;\;\;\;\;\texttt{Layer}^{(2)}(\\
&\;\;\;\;\;\texttt{Layer}^{(1)}(\mathbf{x}))\ldots)

\end{align}
```

* where $\texttt{Layer}(\mathbf{x}) = \sigma.(\texttt{Linear}(\mathbf{x};  \mathbf{W}, \mathbf{b}))$
* ``L`` function compositions
""", md"""

### Show me the code


```python

# create a list of layers of Linear + activations

Layers = [Linear(in_dim, h1), 
		  ReLu(), 
		  Linear(h1, h2), 
		  ReLu(), ..., 
		  Linear(hL, ho)]

# forward computation: function composition
output = x
for layer in layers:
	output = layer(output)
```
""")

# ╔═╡ 1e2db1d6-e487-4783-ae6c-b230f8566732
# md"""

# ## Question


# !!! question "Question"
# 	What if only linear activation functions are used? 
#     * for example, identity activation functions are used only
# 	```math
# 		\sigma(z) = z
# 	```

# """

# ╔═╡ 24d1beda-717c-4f5b-8220-d8671dfc8187
# Foldable("Answer", md"""

# The neural network reduces to a linear function at the end!

# With identity activation, a dense layer is just a few hyperplanes (to be more mathematically correct, affine transforms)

# ```math

# \texttt{Dense}(\mathbf{x}) = \mathbf{W} \mathbf{x} +\mathbf{b}
# ```

# Consider a NN with 2 layers: composing them together is just another set of hyperplanes

# ```math
# \begin{align}
# \texttt{nnet}(\mathbf{x}) &= \texttt{Dense}^{(2)}(\texttt{Dense}^{(1)}(\mathbf{x})) \\
# &= \texttt{Dense}^{(2)}(\mathbf{W}^{(1)} \mathbf{x} +\mathbf{b}^{(1)})\\
# &= \mathbf{W}^{(2)} (\mathbf{W}^{(1)} \mathbf{x} +\mathbf{b}^{(1)}) + \mathbf{b}^{(2)}\\
# &= \underbrace{\mathbf{W}^{(2)}\mathbf{W}^{(1)}}_{\tilde{\mathbf{W}}} \mathbf{x} +\underbrace{\mathbf{W}^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)}}_{\tilde{\mathbf{b}}}\\
# &= \tilde{\mathbf{W}} \mathbf{x} + \tilde{\mathbf{b}}
# \end{align}
# ```

# Adding more layers do not change a thing, at the end still a linear function!

# """)

# ╔═╡ 495ff258-ba4a-410e-b5e5-30aba7aaa95e
md"""

## Implementation -- MLP
##### _be classy_

"""

# ╔═╡ 1d5accd1-3582-4c76-8680-30e34d27142b
md"""
```python
class MLP:
    def __init__(self, layers):
        self.layers = layers
  
    def __call__(self, x):
        yhat = x
        for layer in self.layers:
            yhat = layer(yhat)
        return yhat
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

"""

# ╔═╡ 0be45c76-6127-48e8-95fa-28827de2f38e
TwoColumn(md"""

For example, a MLP like this: 

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp6.svg", :width=>300))

""", 
md"""
Implementation:
\
\

```python
n_i, n_h, n_o = 2, 2, 1 ## number of input and hidden units

## instantiate a MLP with ReLu activation
nnet = MLP([Linear(n_i, n_h), ReLu(), Linear(n_h, n_o)])
output = nnet(x)
```

"""

)

# ╔═╡ 5c7370bc-7ba6-4449-b963-448283c80315
nn1 = let
	Random.seed!(111)
	input_dim, hidden_dim, out_dim = 2, 2, 1
	l1 = DenseLayer(input_dim, hidden_dim, tanh)
	l2 = DenseLayer(hidden_dim, out_dim, σ)
	neural_net(x) = (l2 ∘ l1)(x)
end;

# ╔═╡ 63937001-c4d9-423f-9491-b4f35342f5a4
# let
# 	i = 1
# 	nn1(D[i, :])
# end

# ╔═╡ e259603f-2baa-4242-bc04-791d1c8b168e
# nn1(D')

# ╔═╡ bedac901-af79-4798-b7b3-c9a730220351
# md"""

# ## *Why transpose

# Note that

# ```math
# \mathbf{X}=\begin{bmatrix}(\mathbf{x}^{(1)})^\top \\ (\mathbf{x}^{(2)})^\top \\ \vdots \\  (\mathbf{x}^{(n)})^\top \end{bmatrix}

# ```

# * apply transpose ``^\top`` to ``\mathbf{X}``: the columns become the observations

# ```math
# \mathbf{X}^\top=\begin{bmatrix}\mathbf{x}^{(1)}& \mathbf{x}^{(2)} &\ldots& \mathbf{x}^{(n)}\end{bmatrix}

# ```
# * consider one layer 

# ```math

# \mathbf{W} \underbrace{\begin{bmatrix}\mathbf{x}^{(1)}& \mathbf{x}^{(2)} &\ldots& \mathbf{x}^{(n)}\end{bmatrix}}_{\mathbf{X}^\top} +\mathbf{b} = \begin{bmatrix}\mathbf{W}\mathbf{x}^{(1)}+\mathbf{b}&  \mathbf{W}\mathbf{x}^{(2)}+\mathbf{b} &\ldots& \mathbf{W}\mathbf{x}^{(n)}+\mathbf{b}\end{bmatrix} 
# ```


# * the i-th column is the output of the i-th observation
# """

# ╔═╡ eeb9e21e-d891-4b7b-a92a-d0d0d70c1517
# md"""

# ## Implementation -- NN

# Neural net is just composition of layers of different constructs

# * *e.g.* `DenseLayer`

# Composing functions together is simple in Julia 

# * just use ``∘`` (just type "\circ" + `tab`): the same way as you write maths!


# ```julia
# (f ∘ g)(x) # composing functions together; the same as f(g(x))
# ```
# """

# ╔═╡ 15604ca0-b8c0-433b-a15c-3818c8ced94a
md"""

## Multi-output problems



For hand digit recognition problems (MNIST dataset), the target ``y\in \{0,1,2,\ldots, 9\}``
* 10 categories!


It can be easily handled by neural network

* ##### output layer have 10 outputs instead of 1


For example, for a **three** class classification

* the output layer outputs 3 outputs

"""

# ╔═╡ 951a1b38-f53d-41ea-8d41-7a73f4d850fa
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-15.png
' width = '500' /></center>"

# ╔═╡ 5af83948-c7dd-46b5-a920-4bfb514e4f9c
md"""

## Multi-class classification: softmax function


To idea is similar to softmax regression, we apply softmax function to the final output

"""

# ╔═╡ 8243b153-0d03-4efa-9fcc-98ab42008826
aside(tip(md"""

It is a *soft* version of a *hardmax* function:

```math
\begin{bmatrix}
1.3\\
5.1\\
2.2\\
0.7\\
1.1
\end{bmatrix} \Rightarrow \texttt{hardmax} \Rightarrow \begin{bmatrix}
0\\
1\\
0\\
0\\
0
\end{bmatrix},
```

also known as winner-take-all. 

Mathematically, each element is: ``I(\mathbf{a}_i = \texttt{max}(\mathbf{a}))``

"""))

# ╔═╡ 4514df4f-0fee-4e30-8f43-d68a73a56105
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/softmax.jpeg
' width = '500' /></center>"

# ╔═╡ 271874ee-3db9-4abd-8fb4-96f4266cec25
md"figure source: [^1]"

# ╔═╡ 65efc486-8797-442f-a66e-32da5b860635
# function soft_max_naive(x)
# 	ex = exp.(x)
# 	ex ./ sum(ex)
# end

# ╔═╡ b517aaba-e653-491c-8dc2-af86a300b62e
md"""

## NNs for multiclass classification

To put it together, a NN for a 3-class classification looks like this



"""

# ╔═╡ 9d0e9137-f53b-4329-814e-20e842033f41
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnmulticlass.png
' width = '600' /></center>"

# ╔═╡ 1287ed6d-8f1e-40f3-8d46-00ab4b267681
  
md"""figure source [^2]"""

# ╔═╡ 6a378bb3-a9b9-4d94-9407-b5020fefcf39
md"""

## Cross entropy loss (multiclass classification)

Multiclass cross entropy loss

$$
\begin{align}
L(\mathbf{y}; \hat{\mathbf{y}}) &= - \sum_{j=1}^C  \mathbf{y}_j \ln \underbrace{P(y =j| \mathbf{x})}_{\text{NNs softmax: } \hat{\mathbf{y}}_j}\\
&=- \sum_{j=1}^C  \mathbf{y}_j \ln \hat{\mathbf{y}}_j
\end{align}$$

- ``\mathbf{y}`` is the one-hot encoded label
- ``\hat{\mathbf{y}}`` is the softmax output
- the binary case is just a specific case




"""

# ╔═╡ 231e0908-a543-4fd9-80cd-249961a8ddaa
md"""

## Learning 


The same idea: minimise the loss or equivalently maximise the likelihood


Denote the parameter together ``\boldsymbol{\Theta} = \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, \mathbf{b}^{(2)}, \ldots\}``

```math
\hat{\boldsymbol{\Theta}} \leftarrow \arg\min_{\boldsymbol{\Theta}}\frac{1}{n} \sum_{i=1}^n L(y^{(i)}, \hat{y}^{(i)}) 
```

* where ``\hat{y}^{(i)}`` is the output of the neural network


##### Cross-entropy loss for binary classification

"""

# ╔═╡ 188654a1-5211-4f87-ae1c-7a64233f8d28
(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp7.svg", :width=>700))


# ╔═╡ cd5c4feb-8da9-41f3-8a71-430eb254fb95
md"""

##### MSE loss for regression

"""

# ╔═╡ 14482e14-e64c-4b78-abf4-469f1c404776
(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg_loss.svg", :width=>700))


# ╔═╡ dcaa1b2d-6f6f-4635-ae39-06a9e9555bce
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-06.png
# ' width = '640' /></center>"

# ╔═╡ 44833991-5520-4046-acc9-c63a9700acf3
md"""


## Gradient calculation - Backpropagation





**Backpropagation** (BP) is an efficient algorithm to compute the gradient for NN
* a specific case of reverse-mode auto-differentiation algorithm
* check the appendix for details of BP algorithm as well as a BP implementation


##### We will talk about the gradient calculation in detail next time

"""

# ╔═╡ fd1b3955-85eb-45a4-9798-7d453f1cdd28
md"""


## Gradient calculation - Backpropagation





**Backpropagation** (BP) is an efficient algorithm to compute the gradient for NN
* a specific case of reverse-mode auto-differentiation algorithm
* check the appendix for details of BP algorithm as well as a BP implementation


##### We will talk about the gradient calculation next time
\

##### For now

For Python, we use `PyTorch` to compute the gradient

* `PyTorch`'s `backward()` implements the backpropagation 


For Julia, we use `Zygote.jl`
* `Zygote.jl` is a reverse mode auto-diff package
* it also implements the backpropagation behind the scene
"""

# ╔═╡ 231fd696-89f3-4bf8-9466-ddcea3789d21
md"""

## Demonstrations

"""

# ╔═╡ 41b78734-0fae-4323-9af1-e5e0deda584c
md"""

### Create the NN

"""

# ╔═╡ 8a8c1903-45c9-455f-a02a-fa858ae7bada
TwoColumn(md"""


* input size: 2
* hidden size: 2 (activation function e.g. `logistic`)
* output size: 1 (binary classification)
  * logistic output
""", 
	
	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-05.png
' width = '400' /></center>")

# ╔═╡ f892235e-50cc-4d74-b8cc-61768450a9e3
begin
	Random.seed!(111)
	input_dim, hidden_dim, out_dim = 2, 2, 1
	l1 = DenseLayer(input_dim, hidden_dim, logistic)
	l2 = DenseLayer(hidden_dim, out_dim, logistic)
	neural_net(x) = (l2 ∘ l1)(x)
end;

# ╔═╡ 2fbcf298-13e7-4c1e-be10-de8ca82c9385
md"""

Implement the **gradient descent** algorithm:
"""

# ╔═╡ ec5c193a-33de-48c9-a941-ceffc811596f
function cross_entropy_loss(y, ŷ)
	# deal with boundary cases such as σ = 0 or 1, log(0) gracefully
	# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
	# rather you should use xlogy and xlog1py
	-sum(xlogy.(y, ŷ) + xlog1py.(1 .-y, -ŷ))
end;

# ╔═╡ 5c5538ce-9e0e-4fba-9dfb-efa37cd43b9b
# md"""

# Use Zygote.jl to compute the gradient ``\nabla_\boldsymbol{\Theta}L``
# """

# ╔═╡ 8a654c85-7095-4f91-82d0-2393f90b3aa8
# let
# 	Θ = [l1, l2]
# 	∇g_ = Zygote.gradient(() -> cross_entropy_loss(targets', neural_net(D')), Params(Θ)) # Params(θ) tells Zygote what parameters' gradient we are computing
# 	∇g_[l1], ∇g_[l2]
# end

# ╔═╡ e746c069-f18e-4367-8cdb-7ffaac0f9ace
# md"""

# Helper method for gradient **update**
# * it becomes handy if we have a very deep NN (many layers)
# """

# ╔═╡ b1edd6a5-c00d-4ef5-949f-ce12e9507c58
function update!(layer::DenseLayer, ∇layer, γ)
	layer.W .= layer.W + γ * ∇layer.W
	layer.b .= layer.b + γ * ∇layer.b
end;

# ╔═╡ 596a1ddd-1aaa-4b02-ab87-be0a6c3fbdfd
accuracy(ŷ, y) = mean(ŷ .== y); # helper method for accuracy;

# ╔═╡ 69f26e0f-20c3-4e1d-95db-0c365f272f6d
md"""

## Inspect the learnt hidden neurons



##### hidden layer + output layer is a ordinary logistic regression
* from ``\mathbf{a} \in \mathbb{R}^2`` perspective
"""

# ╔═╡ bff67477-f769-44b2-bd07-21b439eced35
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/logis_mlp.svg
' width = '500' /></center>"

# ╔═╡ 3bfccf63-8658-4c52-b2cf-047fcafec8e7
md"""

## Inspect the learnt hidden neurons



##### hidden layer + output layer is a ordinary logistic regression
* from ``\mathbf{a} \in \mathbb{R}^2`` perspective
"""

# ╔═╡ 47cc554e-e2f8-4e90-84a8-64b628ff4703
TwoColumn(md"""

NN **cleverly** and **automatically** *engineered* good features 


```math
\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix}
```

* the learnt features become **linearly separable** for the output layer

""", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/logis_mlp.svg
' width = '500' /></center>")

# ╔═╡ 93ce5d39-5489-415c-9709-8740a016db06
md"""

## Another example -- the smile data

"""

# ╔═╡ b40fd686-c82b-465c-ad8f-bcea54d62aac
begin
	gr()
	# Visualizing
	scatter(real[1,1:500],real[2,1:500], title="A non-linear classification")
	scatter!(fake[1,1:500],fake[2,1:500], ratio=1, framestyle=:origin)
end

# ╔═╡ 8067e844-14ed-4fb9-b609-ff598d61cf9e
begin
	D_smile = [real'; fake']
	targets_smile = [ones(size(real)[2]); zeros(size(fake)[2])]
end;

# ╔═╡ 18f85862-5b89-4358-8431-db7fdd900b9b
begin
	D_smile_test = [real_test'; fake_test'];
	targets_smile_test = [ones(size(real_test)[2]); zeros(size(fake_test)[2])]
end;

# ╔═╡ 743e1b17-23b9-4a2a-9466-4c0ff03e8888
md"""

### Create the network
"""

# ╔═╡ baca7037-f1eb-4b61-b157-9f5e12523894
md"""

Let's add one more hidden layer

```math
\texttt{nnet}(\mathbf{x}) = (\underbrace{\texttt{Layer}^{(3)}}_{\text{hidden 2 to output}} \circ \underbrace{\texttt{Layer}^{(2)}}_{\text{hidden 1 to hidden 2}} \circ \underbrace{\texttt{Layer}^{(1)}}_{\text{input to hidden 1}}) (\mathbf{x})
```

* where $\texttt{Layer}(\mathbf{x}) = \sigma.(\texttt{Linear}(\mathbf{x};  \mathbf{W}, \mathbf{b}))$
"""

# ╔═╡ 3202e689-edb5-4dd5-b8ac-74a4fd0251e6
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-16.png
' width = '500' /></center>"

# ╔═╡ fbc066bb-90ab-4d9d-8f77-010662290f60
begin
	Random.seed!(123)
	# create a neural network
	hidden_1 = 20
	hidden_2 = 3
	l1_2 = DenseLayer(2, hidden_1, tanh)
	l2_2 = DenseLayer(hidden_1, hidden_2, tanh)
	l3_2 = DenseLayer(hidden_2, 1, σ)
	nnet2(x) = (l3_2 ∘ l2_2 ∘ l1_2)(x)
end;

# ╔═╡ 9b7a4c44-76c0-4627-a226-e43c78141031
loss_smile, anim_smile = let
	gr()
	# learning rate
	γ = 0.0001
	iters = 1000
	losses = zeros(iters)
	layers = [l1_2, l2_2, l3_2]
	anim = @animate for i in 1: iters
		∇gt = Zygote.gradient(() -> cross_entropy_loss(targets_smile[:], nnet2(D_smile')[:] ), Params(layers))
		for l in layers
			update!(l, ∇gt[l], -γ)
		end
		losses[i] = cross_entropy_loss(targets_smile[:], nnet2(D_smile')[:])
		if i % 20 == 1
			ŷ = nnet2(D_smile')[:] .> 0.5
			@info "Iteration, accuracy: ", i , accuracy(targets_smile[:], ŷ)*100
		end

		scatter(real[1,1:100],real[2,1:100], zcolor=nnet2(real)', ratio=1, colorbar=false, framestyle=:origin)
		scatter!(fake[1,1:100],fake[2,1:100],zcolor=nnet2(fake)',legend=false, title="Iteration: "*string(i))
	end every 50
	losses, anim
end

# ╔═╡ 79eab834-9a65-4c94-9466-6e9de387dbca
gif(anim_smile, fps=5)

# ╔═╡ c14d5645-4ccb-413c-ad54-ee9d45706405
begin
	gr()
	plot(-0.7:0.1:.7, -0.25:0.1:1, (x, y) -> nnet2([x, y])[1] < 0.5, alpha=0.4,  st=:contourf, c=:jet, framestyle=:origin)
	scatter!(real[1,1:100],real[2,1:100], c=1, ratio=1, colorbar=false)
	scatter!(fake[1,1:100],fake[2,1:100],c=2,legend=false)
end

# ╔═╡ 298fbc0b-5e94-447b-8a6c-a0533490e8e1
md"""

## Inspect the learnt hidden neurons



##### the last hidden layer + output layer is again an ordinary logistic regression
* from ``\mathbf{a}^{(2)} \in \mathbb{R}^3`` perspective
"""

# ╔═╡ 4590f3b7-031d-4039-b125-945e3dcdcea7
TwoColumn(md"""

NN **cleverly** and **automatically** *engineered* good features again


```math
\mathbf{a}^{(2)} = \begin{bmatrix} a_1^{(2)} \\ a_2^{(2)} \\ a_3^{(2)} \end{bmatrix}
```

* the learnt features become **linearly separable** for the output layer

""", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/smile_logits.svg
' width = '500' /></center>")

# ╔═╡ d12ab2fb-a91d-498a-844f-0148e56110d7
let
	import PlotlyBase
	plotly()
	hidden_output = l2_2(l1_2((D_smile')))
	scatter(hidden_output[1, targets_smile .== 0], hidden_output[2, targets_smile .==0], hidden_output[3, targets_smile .==0], legend=:outerright, label="class 1", markersize=2)
	scatter!(hidden_output[1, targets_smile .== 1], hidden_output[2,targets_smile .==1], hidden_output[3, targets_smile .== 1], label="class 2", xlabel="a1", ylabel="a2", zlabel="a3", size=(500,400), title="The hidden layer's output",markersize=2)
end

# ╔═╡ 0506f7ac-e923-4fc8-84ed-a4ffdfd7b615
md"""

### Final accuracy
"""

# ╔═╡ 7997afc0-0d77-4663-8baf-4f4e38b034eb
md"Training accuracy:"

# ╔═╡ fc9e5101-e945-4366-8e0c-fdd057ded536
accuracy(nnet2(D_smile')[:] .> .5, targets_smile)

# ╔═╡ df66af90-ebc0-4e79-8af4-7d7fe3e3e41a
md"Testing accuracy:"

# ╔═╡ 4d55d0a1-541a-4930-bb44-dd713aedae90
accuracy(nnet2(D_smile_test')[:] .> .5, targets_smile_test)

# ╔═╡ 9b4ae619-2951-4f89-befb-b85411826233
md"""

## Another example: non-linear Regression

"""

# ╔═╡ 39eb062f-f5e9-46f0-80df-df6eab168ebd
sse_loss(y, ŷ) = 0.5 * sum((y .- ŷ).^2) / length(y);

# ╔═╡ 5621b8d4-3649-4a3e-ab78-40ed9fd0d865
begin
	gr()
	scatter(x_input, y_output, label="Observations", title="A non-linear regression dataset")	
	plot!(true_f, lw=2, xlim=[-10, 10], framestyle=:default,  lc=:gray, ls=:dash,label="true function", xlabel=L"x", ylabel=L"y")
end

# ╔═╡ a6c97f1e-bde0-427e-b621-799f4857347d
md"""

### Create the network
"""

# ╔═╡ 3dd87e40-38b8-4330-be87-c901ffb7121a
TwoColumn(md"""
\
\
\

\


A NN with the following configuration
```math
1 \Rightarrow 12 \Rightarrow 1
``` 
""", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-17.png
' width = '350' /></center>")

# ╔═╡ 5d679888-578e-49b1-b4a9-d2616d21badf
md"Choose activation function: $(@bind actfun Select([tanh, relu, sin, logistic, Flux.softplus]))"

# ╔═╡ c94e1c57-1bab-428b-936f-a6c1e2ba8237
begin
	Random.seed!(111)
	# create a NN with one hidden layer and 12 neurons
	hidden_size = 12
	l1_3 = DenseLayer(1, hidden_size, (x) -> actfun(x))
	l2_3 = DenseLayer(hidden_size, 1)
	nnet3(x) = (l2_3 ∘ l1_3)(x)
end

# ╔═╡ 101e52d3-906e-46cd-a55c-a4a13f4d19d1
begin
	x_input_f = Float32.(x_input)
	y_output_f = Float32.(y_output)
	# x_test_f = 
end;

# ╔═╡ 9dd4de90-3417-4f6e-b459-a415f4450d7f
begin
	Random.seed!(123)
	nnet_reg = Chain(Dense(1, hidden_size, actfun), Dense(hidden_size, 1))
end;

# ╔═╡ c94dd5af-85d7-4435-ad85-8c408364df69
let
	gr()
	# layers = [l1_3, l2_3]
	γ = 0.005
	optim = Flux.setup(Descent(γ), nnet_reg)

	anim = @animate for i in 1 : 3000
		# ∇gt = Zygote.gradient(() -> sse_loss(nnet3(x_input')[:], y_output[:]), Params(layers))
		# for l in layers
		# 	update!(l, ∇gt[l], -γ)
		# end
		# l = sse_loss(nnet3(x_input')[:], y_output[:])

		l, grads = Flux.withgradient(nnet_reg) do m
			Flux.mse(m(x_input_f'), y_output_f')
		end
		Flux.update!(optim, nnet_reg, grads[1])
		
		scatter(x_input, y_output, label="Observations", framestyle=:origin)	
		plot!(x_input_f, nnet_reg(x_input_f')[:], label="prediction", lw=2, title="Activation: " * string(actfun)*"; Iteration: "*L"%$(i)"*"; loss: "*L"%$(round(l;digits=2))")
	end every 100
	gif(anim, fps=8)
end

# ╔═╡ e91e3b9e-d130-49d6-b334-f5c99fe39d49
# let
# 	gr()
# 	layers = [l1_3, l2_3]
# 	γ = 0.002
# 	anim = @animate for i in 1 : 3000
# 		∇gt = Zygote.gradient(() -> sse_loss(nnet3(x_input')[:], y_output[:]), Params(layers))
# 		for l in layers
# 			update!(l, ∇gt[l], -γ)
# 		end
# 		l = sse_loss(nnet3(x_input')[:], y_output[:])
# 		scatter(x_input, y_output, label="Observations", framestyle=:origin)	
# 		plot!(x_input, nnet3(x_input')[:], label="prediction", lw=2, title="Activation: " * string(actfun)*"; Iteration: "*L"%$(i)"*"; loss: "*L"%$(round(l;digits=2))")
# 	end every 100
# 	gif(anim, fps=8)
# end

# ╔═╡ 2805fb07-2931-4a29-acc2-d6c17a5dbd97
let
	gr()
	Random.seed!(111)
	# create a NN with one hidden layer and 12 neurons
	hidden_size = 12
	l1_3 = DenseLayer(1, hidden_size, (x) -> relu(x))
	l2_3 = DenseLayer(hidden_size, 1)
	nnet3(x) = (l2_3 ∘ l1_3)(x)
	layers = [l1_3, l2_3]
	γ = 0.004
	anim = @animate for i in 1 : 2000
		∇gt = Zygote.gradient(() -> sse_loss(nnet3(x_input')[:], y_output[:]), Params(layers))
		for l in layers
			update!(l, ∇gt[l], -γ)
		end
		l = sse_loss(nnet3(x_input')[:], y_output[:])
		scatter(x_input, y_output, label="Observations", framestyle=:origin)	
		plot!(x_input, nnet3(x_input')[:], label="prediction", lw=2, title="With ReLu activation: Iteration: "*L"%$(i)"*"; loss: "*L"%$(round(l;digits=2))")
	end every 100
	gif(anim, fps=5)
end

# ╔═╡ bce39bea-830f-4b42-a3c9-46af364dd845
md"""

# Demonstrations (Python)

#### Check `Jupyter notebook`


"""

# ╔═╡ 0b03941d-b3fd-45b1-b0b4-7576004b2676
function soft_max(x)  # the input can be a matrix; apply softmax to each column
	ex = exp.(x .- maximum(x, dims=1))
	ex ./ sum(ex, dims = 1)
end;

# ╔═╡ cdba14b1-28dc-4c43-b51f-895f8fc80143
function cross_en_loss(y, ŷ) # cross entropy for multiple class; add ϵ for numerical issue, log cannot take 0 as input!
	-sum(y .* log.(ŷ .+ eps(eltype(y)))) /size(y)[2]
end;

# ╔═╡ 4bc50945-6002-444b-bf1a-afad4d529a30
md"""

# Appendix

"""

# ╔═╡ c21bb9d9-dcc0-4879-913b-f9471d606e7b
md"""

## Image sources
[^1]: https://miro.medium.com/max/1400/1*ReYpdIZ3ZSAPb2W8cJpkBg.jpeg

[^2]: Machine Learning: A probability perspective; Kevin Murphy 2012

[^3]: https://d33wubrfki0l68.cloudfront.net/8850c924730b56bbbe7955fd6593fd628249ecff/275c5/images/multiclass-classification.png
"""

# ╔═╡ c5beb4c7-9135-4706-b4af-688d8f088e21
md"""

## Data

"""

# ╔═╡ 7fb8a3a4-607e-497c-a7a7-b115b5e161c0
begin
	Random.seed!(123)
	n_= 30
	D1 = [0 0] .+ randn(n_,2)
	# D1 = [D1; [-5 -5] .+ randn(n_,2)]
	D2_1 = [5 5] .+ randn(n_,2)
	D2_2 = [-5 -5] .+ randn(n_,2)
	D2 = [D2_1; D2_2]
	D = [D1; D2]
	targets_ = [repeat(["class 1"], n_); repeat(["class 2"], n_*2)]
	targets = [zeros(n_); ones(n_*2)]
	df_class = DataFrame(x₁ = D[:, 1], x₂ = D[:, 2], y=targets_)
end;

# ╔═╡ 95ed0e5c-1d01-49d1-be25-a751f125eb76
TwoColumn(md"""
\
\
\


* ###### Not separable by _one linear_  boundary
\


* ###### In other words, a single neuron is clearly not enough


""", let

	# θs = range(π/4, π/4 + 2π, 20)
	# r = 5.0
	# ws = r * [cos.(θs) sin.(θs)]

	# anim = @animate for w in eachrow(ws)
	@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Non-linear separable data with single neuron", framestyle=:origin, ratio=1, size=(350,350), titlefontsize=8)
	# plot!(-7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([x, y], w)), c=:jet, st=:contour, alpha=0.5, colorbar=false)
	# end
	# gif(anim, fps = 1)
end)

# ╔═╡ fd7e3596-f567-4fca-a28b-883ada2122f9
TwoColumn(md"""
\
\
\


* ###### Not separable by _one linear_  boundary
\


* ###### In other words, a single neuron is clearly not enough


""", let

	θs = range(π/4, π/4 + 2π, 20)
	r = 5.0
	ws = r * [cos.(θs) sin.(θs)]

	anim = @animate for w in eachrow(ws)
		@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Non-linear separable data with single neuron", framestyle=:origin, ratio=1, size=(350,350), titlefontsize=8)
		plot!(-7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([x, y], w)), c=:jet, st=:contour, alpha=0.5, colorbar=false)
	end
	gif(anim, fps = 1)
end)

# ╔═╡ cf29359e-1a43-46f6-a272-61d19d926f86
p_nl_cls = let
	@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Non-linear separable data")

	w1 = 0.5 * [0, 5, 5]
	# w2 = [-23, -5, -5]

	plot!(-7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([1, x, y], w1)), c=:jet, st=:contour, alpha=0.5)
end;

# ╔═╡ e92f6642-bc77-423c-8071-f2f82b738da1
let
	pl1 = @df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="One neuron", alpha=0.1, label="", framestyle=:origin)

	scatter!(D2_1[:, 1], D2_1[:, 2], c=2, label="")
	scatter!(D1[:, 1], D1[:, 2], c=1, label="")


	pl2 = @df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Another neuron", alpha=0.1, label="",framestyle=:origin)

	scatter!(D2_2[:, 1], D2_2[:, 2], c=2, label="")
	scatter!(D1[:, 1], D1[:, 2], c=1, label="")


	w1 = [-23, 5, 5]
	w2 = [-23, -5, -5]

	plot!(pl1, -7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([1, x, y], w1)), c=:jet, st=:contour)


	plot!(pl2, -7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([1, x, y], w2)), c=:jet, st=:contour)

	plot(pl1, pl2, size=(900,400))
end

# ╔═╡ ae6953c9-c202-43bd-b563-7a982179e053
let
	gr()
	@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Non-linear separable data", ratio=1, size=(400,400), framestyle=:origin)
end

# ╔═╡ 87e4e6b7-881e-4430-b731-5a1990b5d583
losses1 = let
	γ = 0.05 # learning rate
	iters = 100
	losses = zeros(iters)
	for i in 1: iters
		∇gt = Zygote.gradient(() -> cross_entropy_loss(targets[:], neural_net(D')[:]), Params([l1, l2]))
		update!(l1, ∇gt[l1], -γ)
		update!(l2, ∇gt[l2], -γ)
		losses[i] = cross_entropy_loss(targets[:], neural_net(D')[:])
		if i % 1 == 0
			ŷ = neural_net(D')[:] .> 0.5
			@info "Iteration, accuracy: ", i , accuracy(targets[:], ŷ)*100
		end
	end
	losses
end;

# ╔═╡ c1ac246e-a8b3-4eae-adb6-c8fe12e039f2
let
	plotly()
	scatter(D1[:, 1], D1[:, 2], - 0.01 .+ zeros(size(D1)[1]), ms=3, label="class 1")
	scatter!(D2[:, 1], D2[:, 2], -0.01 .+  zeros(size(D2)[1]), ms=3, label="class 2")
	plot!(-6:0.2:6, -6:0.2:6, (x,y) -> l1([x, y])[1], st=:surface, c=:jet, alpha=0.8, colorbar=false, xlabel="x1", ylabel="x2", zlabel="a(x)")
	plot!(-6:0.2:6, -6:0.2:6, (x,y) -> l1([x, y])[2], st=:surface, alpha=0.8,c=:jet)
end

# ╔═╡ 21abc2a8-4270-4e47-9d1b-059d31382c00
let
	gr()
	hidden_output = l1(D')
	scatter(hidden_output[1, targets .== 0], hidden_output[2,targets .==0], legend=:outerright, label="class 1")
	scatter!(hidden_output[1, targets .== 1], hidden_output[2,targets .==1], label="class 2", xlabel=L"a_1", ylabel=L"a_2", size=(500,400), title="The hidden layer's output")
end

# ╔═╡ 150f3914-d493-4ece-b661-03abc8c28de9
begin
	function logistic_loss(w, X, y)
		σ = logistic.(X * w)
		# deal with boundary cases such as σ = 0 or 1, log(0) gracefully
		# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
		# rather you should use xlogy and xlog1py
		-sum(xlogy.(y, σ) + xlog1py.(1 .-y, -σ))
	end
end

# ╔═╡ 2f4a1f66-2293-44c9-b75f-f212c1d522fb
function ∇logistic_loss(w, X, y)
	σ = logistic.(X * w)
	X' * (σ - y)
end

# ╔═╡ 6e52567f-4d86-47f4-a228-13c0ff6910ce
D₂, targets_D₂, targets_D₂_=let
	Random.seed!(123)
	D_class_1 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .+2
	D_class_2 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .-2
	data₂ = [D_class_1; D_class_2]
	D₂ = [ones(size(data₂)[1]) data₂]
	targets_D₂ = [ones(size(D_class_1)[1]); zeros(size(D_class_2)[1])]
	targets_D₂_ = [ones(size(D_class_1)[1]); -1 *ones(size(D_class_2)[1])]
	D₂, targets_D₂,targets_D₂_
end

# ╔═╡ 5c102abf-dca7-48e5-a233-9558559e3fb4
losses, wws=let
	# a very bad starting point: completely wrong prediction function
	ww = [0, -5, -5]
	γ = 0.02
	iters = 2000
	losses = zeros(iters+1)
	wws = Matrix(undef, 3, iters+1)
	losses[1] = logistic_loss(ww, D₂, targets_D₂)
	wws[:, 1] = ww 
	for i in 1:iters
		gw = ∇logistic_loss(ww, D₂, targets_D₂)
		# Flux.Optimise.update!(opt, ww, -gt)
		ww = ww - γ * gw
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, D₂, targets_D₂)
	end
	losses, wws
end;

# ╔═╡ ed15aaea-712b-4dcf-89d4-224e81549135
TwoColumn(let
	gr()
	plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label=L"y^{(i)} = 1", xlabel=L"x_1", ylabel=L"x_2", c=2, size=(350,350))
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label=L"y^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6])
end, 
	
	let
	gr()

	anim = @animate for t in [1:25...]
		plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), st=:scatter, label="class 1", c=2, size=(350,350))
		plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], zeros(sum(targets_D₂ .== 0)), st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
		w₀, w₁, w₂ = (wws[:, t])
		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="Iteration: "*L"%$(t);"*" loss: " *L"%$(round(losses[t]; digits=1))", ratio=1)
	end

	gif(anim, fps=5)
end)

# ╔═╡ ff665805-7e65-4fcb-bc0a-0ab323b595f9
let
	gr()
	plot(losses[1:20], label="Loss", xlabel="Epochs", ylabel="Loss", title="Learning: loss vs epochs")
end

# ╔═╡ 48847533-ca98-43ce-be04-1636728becc4
# as: arrow head size 0-1 (fraction of arrow length)
# la: arrow alpha transparency 0-1
function arrow3d!(x, y, z,  u, v, w; as=0.1, lc=:black, la=1, lw=0.4, scale=:identity)
    (as < 0) && (nv0 = -maximum(norm.(eachrow([u v w]))))
    for (x,y,z, u,v,w) in zip(x,y,z, u,v,w)
        nv = sqrt(u^2 + v^2 + w^2)
        v1, v2 = -[u,v,w]/nv, nullspace(adjoint([u,v,w]))[:,1]
        v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
        v5 = v4 - 2*(v4'*v2)*v2
        (as < 0) && (nv = nv0) 
        v4, v5 = -as*nv*v4, -as*nv*v5
        plot!([x,x+u], [y,y+v], [z,z+w], lc=lc, la=la, lw=lw, scale=scale, label=false)
        plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], [z+w,z+w-v5[3]], lc=lc, la=la, lw=lw, label=false)
        plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], [z+w,z+w-v4[3]], lc=lc, la=la, lw=lw, label=false)
    end
end

# ╔═╡ f119f829-51c9-4c15-b0a2-6bbd1db5a428
let
	plotly()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:zerolines, xlabel="x₁", ylabel="x₂", title="h(x) = wᵀ x")
	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel="x₁", ylabel="x₂", title="σ(wᵀx)")

	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	plot(p1, p2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
PlotlyBase = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
DataFrames = "~1.4.3"
Distributions = "~0.25.79"
Flux = "~0.13.8"
HypertextLiteral = "~0.9.4"
Images = "~0.26.0"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.17"
LogExpFunctions = "~0.3.26"
PlotlyBase = "~0.8.19"
Plots = "~1.36.3"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.48"
StatsPlots = "~0.15.5"
Zygote = "~0.6.49"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "d2b06af12d44deb09ee3d57f12db9a55253efc61"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "89e0654ed8c7aebad6d5ad235d6242c2d737a928"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.3"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "968c1365e2992824c3e7a794e30907483f8469a9"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "4.4.1"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "498f45593f6ddc0adff64a9310bb6710e851781b"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.5.0+1"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "bcc4a23cbbd99c8535a5318455dcf0f2546ec536"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "5248d9c45712e51e27ba9b30eebec65658c6ce29"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.6.0+0"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "75923dce4275ead3799b238e10178a68c07dbd3b"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "8.9.4+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "f98ae934cd677d51d2941088849f0bf2f59e6f6e"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.53.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "b86ac2c5543660d238957dbde5ac04520ae977a7"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.4"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "a1296f0fe01a4c3f9bf0dc2934efbf4416f5db31"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "d9a8f86737b665e15a9641ecbac64deef9ce6724"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.23.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fe2838a593b5f776e1597e086dcd47560d94e816"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.3"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d4f69885afa5e6149d0cab3818491565cf41446d"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.4.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "b6def76ffad15143924a2199f72a5cd883a2e8a9"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.9"
weakdeps = ["SparseArrays"]

    [deps.Distances.extensions]
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "938fe2981db009f531b6332e31c58e9584a2f9bd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.100"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "a20eaa3ad64254c61eeb5f230d9306e937405434"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.6.1"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote", "cuDNN"]
git-tree-sha1 = "3e2c3704c2173ab4b1935362384ca878b53d4c34"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.17"

    [deps.Flux.extensions]
    AMDGPUExt = "AMDGPU"
    FluxMetalExt = "Metal"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9a68d75d466ccc1218d0552a8e1631151c569545"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "2e57b4a4f9cc15e85a24d603256fe08e527f48d1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.8.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "72b2e3c2ba583d1a7aa35129e56cf92e07c083e3"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.21.4"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "4423d87dc2d3201f3f1768a29e807ddc8cc867ef"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3657eb348d44575cc5560c80d7e55b812ff6ffe1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.8+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1cf1d7dcb4bc32d7b4a5add4232db3750c27ecb4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.8.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "f5356e7203c4a9954962e3757c08033f2efe578a"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.0"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "fc5d1d3443a124fde6e92d0260cd9e064eba69f8"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.1"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "432ae2b430a18c58eb7eca9ef8d0f2db90bc749c"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.8"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "b0b765ff0b4c3ee20ce6740d843be8dfce48487c"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.3.0"

[[deps.ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "3ff0ca203501c3eedde3c6fa7fd76b703c336b5f"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.2"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "7ec124670cbce8f9f0267ba703396960337e54b5"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.0"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "d438268ed7a665f8322572be0dabda83634d5f45"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.0"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "8e59ea773deee525c99a8018409f64f19fb719e6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.7"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "aa6ffef1fd85657f4999030c52eaeec22a279738"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.33"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "327713faef2a3e5c80f96bf38d1fa26f7a6ae29e"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "81dc6aefcbe7421bd62cb6ca0e700779330acff8"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.25"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "4c5875e4c228247e1c2b087669846941fb6e0118"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.8"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "8695a49bfe05a2dc0feeefd06b4ca6361a018729"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.1.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c35203c1e1002747da220ffc3c0762ce7754b08c"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.23+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "c88a4afe1703d731b1c4fdf4e3c7e77e3b176ea2"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.165"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "3504cdb8c2bc05bde4d4b09a81b01df88fcbbba0"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.3"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "72240e3f5ca031937bd536182cb2c031da5f46dd"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.21"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics", "cuDNN"]
git-tree-sha1 = "f94a9684394ff0d325cc12b06da7032d8be01aaf"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.7"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "5e4029759e8699ec12ebdf8721e51a659443403c"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.4"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "bbb5c2115d63c2f1451cb70e5ef75e8fe4707019"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.22+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "c1fc26bab5df929a5172f296f25d7d08688fd25b"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.20"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "9b02b27ac477cad98114584ff964e3052f656a0f"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "6a9521b955b816aa500462951aa67f3e4467248a"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.36.6"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3aa2bb4982e575acd7583f01531f241af077b163"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.13"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "ee094908d720185ddbdc58dbe0c1cbe35453ec7a"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "ae36206463b2395804f2787ffe172f44452b538d"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.8.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "da095158bdc8eaccb7890f9884048555ab771019"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "1e597b93700fa4045d7189afa7c004e0584ea548"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "54ccb4dbab4b1f69beb255a2c0ca5f65a9c82f08"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.5.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "4b8586aece42bee682399c4c4aee95446aa5cd19"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.39"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "03fec6800a986d191f64f5c0996b59ed526eda25"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.1"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "9cabadf6e7cd2349b6cf49f1915ad2028d65e881"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.2"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "8621f5c499a8aa4aa970b1ae381aae0ef1576966"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.4"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "53bd5978b182fa7c57577bdb452c35e5b4fb73a5"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.78"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "e2fe78907130b521619bc88408c859a472c4172b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.63"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDNN_jll"]
git-tree-sha1 = "ee79f97d07bf875231559f9b3f2649f34fac140b"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.1.0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─ab7b3ac3-953d-41fb-af70-7e92ab06b8c7
# ╟─28754109-9d20-4c20-a2b1-491dd56dcc2f
# ╟─50103dcc-034e-4c22-953c-1b5fd781c070
# ╟─f72cacd8-f5c2-43a7-b737-71c6b5185880
# ╟─fdab762e-4d32-43f9-8a0d-267fec8e491e
# ╟─78aad856-dc62-430d-ad3a-a32b0d268b0d
# ╟─a9c35a78-c179-45f4-ba25-18e744b30129
# ╟─1e180936-7d39-4895-b526-9a09d7024946
# ╟─16a460dd-a1bb-4304-bd80-cdeaa82c1cac
# ╟─70d6cadd-04e4-4bb5-b38e-8d510a3f2bcf
# ╟─f413ae69-9df1-4447-84b0-8f35b39f55cd
# ╠═5262a879-a236-44ee-8d00-ef8547de579b
# ╟─f119f829-51c9-4c15-b0a2-6bbd1db5a428
# ╟─78d6618f-cd68-434b-93e4-afd36c9e3247
# ╟─0d0b78f8-8b53-4f11-94d9-d436c84ae976
# ╟─a7b797a8-d791-4d10-9ac2-7bd9c0815688
# ╟─bbb08fad-6d96-4735-9ac5-a03f27773af1
# ╟─7159fa02-e801-4a86-8653-aae1dda2e7ac
# ╟─f697e1bb-4e8b-458c-8ea1-d85c403fd913
# ╟─0b945b04-2808-4e81-9c2f-18ef5c891366
# ╟─f510ba7f-8ca5-42ed-807d-67a8710d2e00
# ╟─45788db8-4d37-4b7a-aadb-6c6993bf37e8
# ╟─ed15aaea-712b-4dcf-89d4-224e81549135
# ╟─5c102abf-dca7-48e5-a233-9558559e3fb4
# ╟─f000ec11-e922-4262-b985-3882b2e6c2ee
# ╟─c95275ed-b682-49b8-ae54-9e9e84d401e6
# ╟─1d968144-0bf7-4e08-9600-6fc33cdbdb52
# ╟─7dcff9b2-763b-4424-97d7-48da1e3d2aff
# ╟─6d84e048-d215-4597-b1f9-dbc24e031791
# ╟─2945e298-1983-4e6b-a903-bead1ac833a8
# ╟─1fb81ca3-2fa9-4862-bad1-00446515a23a
# ╟─95ed0e5c-1d01-49d1-be25-a751f125eb76
# ╟─35054345-67ff-4a5f-8b01-7f6ddf087fcc
# ╟─fd7e3596-f567-4fca-a28b-883ada2122f9
# ╟─cf29359e-1a43-46f6-a272-61d19d926f86
# ╟─fe45432f-9224-4e1e-a981-0136db240085
# ╟─e92f6642-bc77-423c-8071-f2f82b738da1
# ╟─6886fe8d-9645-4094-85f8-6b3fb5d409e4
# ╟─f52e374a-2031-4bd9-b58c-16598ffac15a
# ╟─4b2fd42a-e88d-4117-af79-84cfaf902122
# ╟─5bdd174d-9250-4864-80af-1360240ccc93
# ╟─36a00007-0da1-48a4-a225-d400d9c61a37
# ╟─ca75fcaf-9b6d-4d1d-a97e-1b0167b1e2d8
# ╟─f6e77a11-8e1a-44e2-a2d7-cd29939dcc30
# ╟─001a7866-f0c4-4cea-bda3-d5fea2ce56b4
# ╟─2271ac6d-4bb7-4683-8672-c280c3e57fd3
# ╟─350146ba-6e11-41f8-b451-b3c842722de9
# ╟─fcb991b7-e3e1-4f82-8a8b-c1f2a99d6cce
# ╟─8971420b-7687-4a89-8cfa-da2483be8730
# ╟─a7051da0-f28a-4261-b71c-f7450341e1bf
# ╟─500d6f5f-30ee-45a0-9e3c-5f8ba5274e75
# ╟─2ec97a4c-0ddc-4918-91a1-0bf177b4a202
# ╟─b7d84cb8-59ee-41e6-a04e-a4be9f905f95
# ╟─d6023ad2-2467-44a4-8c57-d12024e76536
# ╟─82a2a13c-cdbe-4901-8b2e-86a5d3cd99e5
# ╟─afce64cb-cdca-42a1-8e92-f1e9634af976
# ╟─7841a726-072a-41a3-a10e-4a4a5256aef1
# ╠═ddb1bd96-e316-450a-a7b5-f5561a890266
# ╠═0fe62e04-559d-4140-970b-6baca845be63
# ╟─8d009757-ab79-47e5-8be0-080ef091b302
# ╟─f079377b-d71f-4f70-aece-f04d12cd3127
# ╟─fe63494b-716c-4718-afcd-026291dc7b94
# ╟─e6d3f1d1-21cb-4662-8806-77b88038be7c
# ╟─cb1bc71b-8c8b-4873-8f28-2935c0972b8c
# ╟─4a0195bf-2d5c-478b-b571-edd4b3e532eb
# ╟─13786181-9280-4260-9705-f2aa419e9c26
# ╟─fec26acd-d545-428d-a90d-17a89b2e11c9
# ╟─6d443b0c-becc-459a-b4ee-9a863c7d69b3
# ╟─be7087be-f948-4701-b5c2-00580ca27d9b
# ╟─c98638f4-5bb1-4550-aa51-945c4632b801
# ╟─eef80655-c86f-4947-be7e-ec5b7ff13ad2
# ╟─8aa1f78a-9cf8-4186-b9a0-31b8a00eabfc
# ╟─81c2fa7b-d3b3-4fe3-b2e6-831a10616ed0
# ╟─361d76ed-7952-4c64-830c-27c6273272d7
# ╟─7001dfa7-fb39-40fc-acac-750c7f0d6605
# ╟─743030fd-4462-4563-8767-2d4bfbb17f5b
# ╟─f3ca219a-e2af-45f7-85bf-3c050227a33b
# ╠═19142836-f9c0-45c1-a05b-24d810a06f8f
# ╟─a5efe47c-71fb-4b1e-8417-47036b96238d
# ╟─bd6ccedb-00a7-4f9d-8ec9-861d8a8ada11
# ╟─378c520d-881c-4056-a3d0-03fcddf2126c
# ╟─dbddfc50-5d66-45c9-b61a-a554282e38c7
# ╟─0f6a50ea-6165-4e9b-b332-641fed50eea9
# ╟─bad0d1a7-f93a-4af8-a453-4ea8e1385d7c
# ╟─c6de70e6-14f4-4b1a-8097-4e086a7fedde
# ╟─bb317788-8d04-45f8-84f7-7a0226a7df6f
# ╟─939c8687-3620-4e98-b9f5-0c304312ef2f
# ╟─a617405e-fc19-4032-8642-7f01fb486c74
# ╟─a8f32a69-3150-4639-9cd4-546ab8442d90
# ╟─6daa681e-3974-497c-baec-347740b2a29e
# ╟─3db05083-dc9e-49d9-9d11-68562eac5827
# ╟─8674f942-dfae-4891-b3e1-a38b79a2b621
# ╟─1be45d72-0661-4a3a-875f-888d0c12b5e4
# ╟─d6353c42-b07b-483a-a669-857c480385b9
# ╟─0e929a8d-2c43-4e51-b990-8855963b4b4d
# ╟─0fe8412f-4dc8-4698-b434-81763038e768
# ╟─6a4faffd-39e9-4576-a820-3f3abb724055
# ╟─a9427f86-20d2-43ca-b1ca-9724252315da
# ╟─afe83e7a-5e5f-4fe4-aa82-082c7fcdfed3
# ╟─13e0cd61-2f3f-40be-ad4a-7a2fb13661bf
# ╟─a594cfb0-bd4d-427e-ba00-c665a2c41f54
# ╟─9a97cc7a-3b3f-41bf-a41a-80dde51a01cc
# ╟─9907c198-8f88-46cf-804d-ba6bde31770b
# ╟─61873a36-2262-4a3b-bb2d-83417c76914a
# ╟─3fde5353-863b-4fd0-8ce2-c6c26d934a14
# ╟─cf69619e-9313-4d14-826b-9a249e1e6b06
# ╟─2c42c70f-05d9-4d6e-8ddd-d225d65a39a6
# ╟─e1a62d74-aaae-491e-b5ef-89e3c1516ca9
# ╟─8b66daf7-4bf7-4323-be48-a6defefcf022
# ╟─91deb3a5-5a59-4a71-a5f0-8037024e1e97
# ╟─1fe38da7-09ca-4006-8c18-97f4a5c5ce78
# ╟─9dea643a-9f72-404b-b57f-496933cc8013
# ╟─73b965c6-4f67-4130-ba81-e72b809b00b6
# ╟─b5a49d08-1786-405b-8b6c-17a9ead69cc2
# ╟─5e7d2395-0095-4c3f-bf21-b2c99fa34e4f
# ╟─f3f3036f-2744-47bc-95a6-6512460023f7
# ╟─f815155f-7d28-473d-8fa9-e0e7f7d81df3
# ╟─ca91ae11-7b2f-4b83-a06c-e66a51ec53a7
# ╟─5159ff24-8746-497b-be07-727f9b7a4d82
# ╟─6a20011e-c89b-46b4-a1f9-243bd401c8fa
# ╟─0fbac3bc-3397-49f6-9e9d-1e31908f530e
# ╟─08ed93ed-304a-46a5-bfc3-c8f45df246ed
# ╟─96e3e41c-0c50-4e9b-b308-2b307fe54fc8
# ╟─1e2db1d6-e487-4783-ae6c-b230f8566732
# ╟─24d1beda-717c-4f5b-8220-d8671dfc8187
# ╟─495ff258-ba4a-410e-b5e5-30aba7aaa95e
# ╟─1d5accd1-3582-4c76-8680-30e34d27142b
# ╟─0be45c76-6127-48e8-95fa-28827de2f38e
# ╟─5c7370bc-7ba6-4449-b963-448283c80315
# ╟─63937001-c4d9-423f-9491-b4f35342f5a4
# ╟─e259603f-2baa-4242-bc04-791d1c8b168e
# ╟─bedac901-af79-4798-b7b3-c9a730220351
# ╟─eeb9e21e-d891-4b7b-a92a-d0d0d70c1517
# ╟─15604ca0-b8c0-433b-a15c-3818c8ced94a
# ╟─951a1b38-f53d-41ea-8d41-7a73f4d850fa
# ╟─5af83948-c7dd-46b5-a920-4bfb514e4f9c
# ╟─8243b153-0d03-4efa-9fcc-98ab42008826
# ╟─4514df4f-0fee-4e30-8f43-d68a73a56105
# ╟─271874ee-3db9-4abd-8fb4-96f4266cec25
# ╟─65efc486-8797-442f-a66e-32da5b860635
# ╟─b517aaba-e653-491c-8dc2-af86a300b62e
# ╟─9d0e9137-f53b-4329-814e-20e842033f41
# ╟─1287ed6d-8f1e-40f3-8d46-00ab4b267681
# ╟─6a378bb3-a9b9-4d94-9407-b5020fefcf39
# ╟─231e0908-a543-4fd9-80cd-249961a8ddaa
# ╟─188654a1-5211-4f87-ae1c-7a64233f8d28
# ╟─cd5c4feb-8da9-41f3-8a71-430eb254fb95
# ╟─14482e14-e64c-4b78-abf4-469f1c404776
# ╟─dcaa1b2d-6f6f-4635-ae39-06a9e9555bce
# ╟─44833991-5520-4046-acc9-c63a9700acf3
# ╟─fd1b3955-85eb-45a4-9798-7d453f1cdd28
# ╟─231fd696-89f3-4bf8-9466-ddcea3789d21
# ╟─ae6953c9-c202-43bd-b563-7a982179e053
# ╟─41b78734-0fae-4323-9af1-e5e0deda584c
# ╟─8a8c1903-45c9-455f-a02a-fa858ae7bada
# ╟─f892235e-50cc-4d74-b8cc-61768450a9e3
# ╟─2fbcf298-13e7-4c1e-be10-de8ca82c9385
# ╟─87e4e6b7-881e-4430-b731-5a1990b5d583
# ╟─ff665805-7e65-4fcb-bc0a-0ab323b595f9
# ╟─ec5c193a-33de-48c9-a941-ceffc811596f
# ╟─5c5538ce-9e0e-4fba-9dfb-efa37cd43b9b
# ╟─8a654c85-7095-4f91-82d0-2393f90b3aa8
# ╟─e746c069-f18e-4367-8cdb-7ffaac0f9ace
# ╟─b1edd6a5-c00d-4ef5-949f-ce12e9507c58
# ╟─596a1ddd-1aaa-4b02-ab87-be0a6c3fbdfd
# ╟─69f26e0f-20c3-4e1d-95db-0c365f272f6d
# ╟─bff67477-f769-44b2-bd07-21b439eced35
# ╟─3bfccf63-8658-4c52-b2cf-047fcafec8e7
# ╟─47cc554e-e2f8-4e90-84a8-64b628ff4703
# ╟─c1ac246e-a8b3-4eae-adb6-c8fe12e039f2
# ╟─21abc2a8-4270-4e47-9d1b-059d31382c00
# ╟─93ce5d39-5489-415c-9709-8740a016db06
# ╟─b40fd686-c82b-465c-ad8f-bcea54d62aac
# ╟─8067e844-14ed-4fb9-b609-ff598d61cf9e
# ╟─18f85862-5b89-4358-8431-db7fdd900b9b
# ╟─743e1b17-23b9-4a2a-9466-4c0ff03e8888
# ╟─baca7037-f1eb-4b61-b157-9f5e12523894
# ╟─3202e689-edb5-4dd5-b8ac-74a4fd0251e6
# ╠═fbc066bb-90ab-4d9d-8f77-010662290f60
# ╠═9b7a4c44-76c0-4627-a226-e43c78141031
# ╟─79eab834-9a65-4c94-9466-6e9de387dbca
# ╟─c14d5645-4ccb-413c-ad54-ee9d45706405
# ╟─298fbc0b-5e94-447b-8a6c-a0533490e8e1
# ╟─4590f3b7-031d-4039-b125-945e3dcdcea7
# ╟─d12ab2fb-a91d-498a-844f-0148e56110d7
# ╟─0506f7ac-e923-4fc8-84ed-a4ffdfd7b615
# ╟─7997afc0-0d77-4663-8baf-4f4e38b034eb
# ╟─fc9e5101-e945-4366-8e0c-fdd057ded536
# ╟─df66af90-ebc0-4e79-8af4-7d7fe3e3e41a
# ╟─4d55d0a1-541a-4930-bb44-dd713aedae90
# ╟─9b4ae619-2951-4f89-befb-b85411826233
# ╟─39eb062f-f5e9-46f0-80df-df6eab168ebd
# ╟─5621b8d4-3649-4a3e-ab78-40ed9fd0d865
# ╟─a6c97f1e-bde0-427e-b621-799f4857347d
# ╟─3dd87e40-38b8-4330-be87-c901ffb7121a
# ╟─5d679888-578e-49b1-b4a9-d2616d21badf
# ╟─c94e1c57-1bab-428b-936f-a6c1e2ba8237
# ╟─101e52d3-906e-46cd-a55c-a4a13f4d19d1
# ╟─9dd4de90-3417-4f6e-b459-a415f4450d7f
# ╟─c94dd5af-85d7-4435-ad85-8c408364df69
# ╟─e91e3b9e-d130-49d6-b334-f5c99fe39d49
# ╟─2805fb07-2931-4a29-acc2-d6c17a5dbd97
# ╟─bce39bea-830f-4b42-a3c9-46af364dd845
# ╟─0b03941d-b3fd-45b1-b0b4-7576004b2676
# ╟─cdba14b1-28dc-4c43-b51f-895f8fc80143
# ╟─4bc50945-6002-444b-bf1a-afad4d529a30
# ╟─c21bb9d9-dcc0-4879-913b-f9471d606e7b
# ╟─c5beb4c7-9135-4706-b4af-688d8f088e21
# ╠═7fb8a3a4-607e-497c-a7a7-b115b5e161c0
# ╠═150f3914-d493-4ece-b661-03abc8c28de9
# ╠═2f4a1f66-2293-44c9-b75f-f212c1d522fb
# ╠═6e52567f-4d86-47f4-a228-13c0ff6910ce
# ╠═48847533-ca98-43ce-be04-1636728becc4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
