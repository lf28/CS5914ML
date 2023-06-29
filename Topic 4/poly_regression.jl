### A Pluto.jl notebook ###
# v0.19.26

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

# ╔═╡ 17a3ac47-56dd-4901-bb77-90171eebc8c4
begin
	using PlutoTeachingTools
	using PlutoUI
	# using Plots
	using LinearAlgebra
	using StatsPlots
	using LogExpFunctions
	using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	using Statistics
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
end

# ╔═╡ 29998665-0c8d-4ba4-8232-19bd0de71477
begin
	using DataFrames, CSV
	using MLDatasets
	# using Images
end

# ╔═╡ f79bd8ab-894e-4e7b-84eb-cf840baa08e4
using Logging

# ╔═╡ cb72ebe2-cea8-4467-a211-5c3ac7af74a4
TableOfContents()

# ╔═╡ 358dc59c-8d06-4272-9a13-6886cdaf3dd9
ChooseDisplayMode()

# ╔═╡ 9bd2e7d6-c9fb-4a67-96ef-049f713f4d53
md"""

# CS5914 Machine Learning Algorithms


#### Polynormial regression

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ cf9c3937-3d23-4d47-b329-9ecbe0006a1e
md"""

## Notations


Super-index with brackets ``.^{(i)}``: ``i \in \{1,2,\ldots, n\}`` index for observations/data
* ``n`` total number of observations
* *e.g.* ``y^{(i)}`` the i-th observation's label
* ``\mathbf{x}^{(i)}`` the i-th observation's predictor vector

Subscript-index: feature/predictor index ``j \in \{1,2,,\ldots, m\} ``
* ``m`` total number of features
* *e.g.* ``\mathbf{x}^{(i)}_2``: the second entry/predictor/feature of ``i``-th observation vector


Vectors: **Bold-face** smaller case:
* ``\mathbf{x},\mathbf{y}, \boldsymbol{\beta}``
* ``\mathbf{x}^\top``: row vector

Matrices: **Bold-face** capital case: 
* ``\mathbf{X},\mathbf{A}, \boldsymbol{\Gamma}``  


Scalars: normal letters
* ``x,y,\beta,\gamma``

"""

# ╔═╡ dfcfd2c0-9f51-48fb-b91e-629b6934dc0f
md"""

# Polynomial regression

"""

# ╔═╡ c4e497fc-cfbf-4d0b-9a0c-1071f2f43a98
linear_reg_normal_eq(X, y) = X \y;

# ╔═╡ 0e2dc755-57df-4d9a-b4f3-d01569c3fcde
begin
	X_housing = MLDatasets.BostonHousing().features |> Matrix |> transpose
	df_house = DataFrame(X_housing', MLDatasets.BostonHousing().metadata["feature_names"] .|> lowercase)
	df_house[!, :target] = (MLDatasets.BostonHousing().targets |> Matrix )[:]
end;

# ╔═╡ c4f42980-0e68-4943-abbe-b28f05dd3ee5
function loss(w, X, y) # in matrix notation
	error = y - X * w
	0.5 * dot(error, error)
end;

# ╔═╡ 8b479e5c-8151-4769-9d27-b5661285b497
md"""

## Recap: _linear_ regression 
"""

# ╔═╡ 55fbb81b-8323-4405-beb9-acd557e6d9f3
TwoColumn(md"""
\
\

!!! note "Linear regression"
	**Linear regression**: ``h(\cdot)`` is assumed **linear**

	```math
	\Large
	h(x_{\text{room}}) = w_0 + w_1 x_{\text{room}} 
	```


""", let
	@df df_house scatter(:rm, :target, xlabel="room", ylabel="price", label="", title="Linear regression", size=(350,300))
	x_room = df_house[:, :rm]
	# x_room_sq = x_room.^2
	X_train_room = [ones(length(x_room)) x_room]
	c, b = linear_reg_normal_eq(X_train_room, df_house.target)

	
	plot!(3.5:0.5:9, (x) -> b* x+ c, lw=3, label=L"h(x) = w_0 + w_1x")
end)

# ╔═╡ 86f09ee8-087e-47ac-a81e-6f8c38566774
md"""

## Non-linear ``h(\mathbf{x})``


**However**, real life data-relationship is **rarely linear**



_For example_, a quadratic ``h(x)`` fits the house data **better**

```math
\Large
\colorbox{pink}{$h(x_{\rm room}) = w_0 + w_1 x_{\rm room} + w_2 x_{\rm room}^2$}\tag{Quadratic}
```

```math
\Large
vs
```

```math
\Large
\colorbox{lightgreen}{$h(x_{\rm room}) = w_0 + w_1 x_{\rm room}$} \tag{Linear}
```
"""

# ╔═╡ f8f34671-ef4c-4300-a983-10d42d43fb9f
quadratic_fit = let
	gr()
	@df df_house scatter(:rm, :target, xlabel="room", ylabel="price", label="", title="House price prediction", size=(400,400))
	x_room = df_house[:, :rm]
	x_room_sq = x_room.^2
	X_train_room = [ones(length(x_room)) x_room x_room_sq]
	X_train_room_ = [ones(length(x_room)) x_room]
	c, b, a = linear_reg_normal_eq(X_train_room, df_house.target)
	c_, b_ = linear_reg_normal_eq(X_train_room_, df_house.target)
	sse = loss([c, b, a] , X_train_room, df_house.target)/ length(df_house.target)
	sse_ = loss([c_, b_] , X_train_room_, df_house.target)/ length(df_house.target)
	plot!(3.5:0.1:9, (x) -> a* x^2 + b* x+ c, lw=3, label=L"h(\mathbf{x}) = w_0+ w_1 x+ w_2 x^2;\;"*L"\mathbf{loss}=%$(round(sse; digits=2))", legend=:outerbottom, legendfontsize=12)
	plot!(3.5:0.5:9, (x) -> b_* x+ c_, lw=3, label=L"h(\mathbf{x}) = w_0 + w_1x;\;"*L"\mathbf{loss}=%$(round(sse_; digits=2))", legend=:outerbottom)
end;

# ╔═╡ 7ab7721d-90e7-4606-8614-f0f5470c9d29
TwoColumn(quadratic_fit, md"""

\
\
\
\

The **M**ean **S**quared **E**rror (MSE) loss: 

| | Linear | Quadratic |
|:---:| :---:| :---:|
|Mean Squared Error| 21.8 | **19.06** |

""")

# ╔═╡ 8fbcf6c3-320c-47ae-b4d3-d710a120eb1a
function least_square_est(X, y) # implement the method here!
	X \ y
end;

# ╔═╡ f65644e7-cb25-46ad-b146-87cf7de69f72
md"""

## Polynomial regression


Quadratic prediction function:

```math
\large
h(x) = w_0 + w_1 x + \boxed{w_2 x^2}
```

Cubic prediction function:

```math
\large
h(x) = w_0 + w_1 x + w_2 x^2 + \boxed{w_3 x^3}
```


A general polynormial regression

```math
\large
h(x) = w_0 + w_1 x + w_2 x^2 + \ldots + w_P x^P =\sum_{p=0}^P w_p x^{p}
```
"""

# ╔═╡ 22cbb1aa-c53f-45d6-891a-90c6f2b9e886
md"""

## Free lunch -- Polynomial regression



For each ``x``， **expand** ``\mathbf{x}`` an additional **engineered** feature ``\colorbox{salmon}{$x^2$}``

* and the expanded input becomes

```math
\large
\tilde{\mathbf{x}} = \begin{bmatrix}1 & x & \colorbox{salmon}{$x^2$}
\end{bmatrix}
```



"""

# ╔═╡ 5148115c-4b3d-4845-97fa-02b68d09ddaa
md"""

## Free lunch -- Polynomial regression



For each ``x``， **expand** with an additional **engineered** feature ``\colorbox{salmon}{$x^2$}``

* and the expanded input becomes

```math
\large
\tilde{\mathbf{x}} = \begin{bmatrix}1 & x & \colorbox{salmon}{$x^2$}
\end{bmatrix}
```


* with learning weight ``\mathbf{w} =[w_0, w_1, w_2]^\top``, the _regression function_ becomes

```math
\large
h(x) =\begin{bmatrix}w_0 & w_1 & w_2\end{bmatrix}  \begin{bmatrix}1 \\ x \\ \colorbox{salmon}{$x^2$} 
\end{bmatrix} = w_0 + w_1 x + \colorbox{salmon}{$w_2x^2$}
```


"""

# ╔═╡ ec41e396-5d30-4b07-9520-6f7a9c5da73b
md"""

## Free lunch -- fixed basis expansion

For each ``x``， **expand** with an additional **engineered** feature ``\colorbox{salmon}{$x^2$}``

* and the expanded input becomes

```math
\large
\tilde{\mathbf{x}} = \begin{bmatrix}1 & x & \colorbox{salmon}{$x^2$}
\end{bmatrix}
```


* for weight ``\mathbf{w} =[w_0, w_1, w_2]^\top``, the _regression function_ becomes

```math
\large
h(x) =\begin{bmatrix}w_0 & w_1 & w_2\end{bmatrix}  \begin{bmatrix}1 \\ x \\ \colorbox{salmon}{$x^2$} 
\end{bmatrix} = w_0 + w_1 x + \colorbox{salmon}{$w_2x^2$}
```

* we can **re-use** the ordinary **least square estimation** result

```math
\mathbf{w}\leftarrow \arg\min_{\mathbf{w}} \frac{1}{2} \underbrace{(\mathbf{y} -\mathbf{h})^\top(\mathbf{y} -\mathbf{h})}_{\text{sum of squared error}}
```


> * this is known as **polynomial regression**

"""

# ╔═╡ dd162f70-73b7-4f1a-beab-6fa26b2b11b1
md"""

## To be more specific

**Expanded input matrix**:

```math
\large 
\tilde{\mathbf{X}} = \begin{bmatrix}1 & x^{(1)} & \columncolor{salmon}(x^{(1)})^2 \\
1 & x^{(2)} & (x^{(2)})^2 \\
\vdots & \vdots & \vdots \\
1 & x^{(n)} & (x^{(n)})^2
\end{bmatrix}

```




"""

# ╔═╡ a495ee86-e78b-46e5-9be4-c4bbfad165f8
md"""

## To be more specific

**Expanded input matrix**:

```math
\large 
\tilde{\mathbf{X}} = \begin{bmatrix}1 & x^{(1)} & \columncolor{salmon}(x^{(1)})^2 \\
1 & x^{(2)} & (x^{(2)})^2 \\
\vdots & \vdots & \vdots \\
1 & x^{(n)} & (x^{(n)})^2
\end{bmatrix}_{n\times 3}

```

The **predictions** ``h(\mathbf{x})`` are

```math
\large
\mathbf{h} = \tilde{\mathbf{X}} \mathbf{w} = \begin{bmatrix}w_0 + w_1 x^{(1)} + w_2(x^{(1)})^2 \\
w_0 + w_1x^{(2)} +w_2 (x^{(2)})^2 \\
\vdots \\
w_0 + w_1x^{(n)} +w_2 (x^{(n)})^2
\end{bmatrix}_{n\times 1}
```


"""

# ╔═╡ d37f5e8f-63a6-440e-b6c0-ae283ded1eb9
md"""

## To be more specific

**Expanded input matrix**:

```math
\large 
\tilde{\mathbf{X}} = \begin{bmatrix}1 & x^{(1)} & \columncolor{salmon}(x^{(1)})^2 \\
1 & x^{(2)} & (x^{(2)})^2 \\
\vdots & \vdots & \vdots \\
1 & x^{(n)} & (x^{(n)})^2
\end{bmatrix}_{n\times 3}

```

The **predictions** are

```math
\large
\mathbf{h} = \tilde{\mathbf{X}} \mathbf{w} = \begin{bmatrix}w_0 + w_1 x^{(1)} + w_2(x^{(1)})^2 \\
w_0 + w_1x^{(2)} +w_2 (x^{(2)})^2 \\
\vdots \\
w_0 + w_1x^{(n)} +w_2 (x^{(n)})^2
\end{bmatrix}_{n\times 1}
```




Lastly, use least square method to estimate ``\mathbf{w}``

```math
\large
\begin{align}
\hat{\mathbf{w}} &\leftarrow \arg\min_{\mathbf{w}} \frac{1}{2}(\mathbf{y} -\mathbf{h})^\top(\mathbf{y}-\mathbf{h})\\
&\;\; \hat{\mathbf{w}} = (\tilde{\mathbf{X}} ^\top\tilde{\mathbf{X}} )^{-1}\tilde{\mathbf{X}}^\top \mathbf{y}

\end{align}
```


"""

# ╔═╡ ce35ddcb-5018-4cb9-b0c9-01fb4b14be40
begin
	x_room = df_house[:, :rm]
	x_room_sq = x_room.^2 # squared x_room^2
	X_room_expanded = [ones(length(x_room)) x_room x_room_sq]
end;

# ╔═╡ 2cdd8751-7ec8-47b0-a174-fdc23e176921
md"""

## Higher orders?


```math
\Large
\boxed{
h(x) = w_0 + w_1 x + w_2 x^2 +\ldots + w_p w^p}
```
 * still _**free lunch**_: regress with a ``n\times \textcolor{red}{(p+1)}`` matrix

"""

# ╔═╡ 720774c4-9aec-4329-bc90-51350fea0191
md"""

```math
\large
\tilde{\mathbf{X}} = \begin{bmatrix}1 & x^{(1)} & (x^{(1)})^2  & \ldots & (x^{(1)})^p \\
1 & x^{(2)} & (x^{(2)})^2 & \ldots &(x^{(2)})^p\\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x^{(n)} & (x^{(n)})^2 & \ldots & (x^{(n)})^p
\end{bmatrix}_{n\times (p+1)}
```
"""

# ╔═╡ 8b596e98-4d0c-471e-bb25-20f492a9199b
md"Polynomial order: $(@bind poly_order Slider(0:10, default =2, show_value=true))"

# ╔═╡ edc245bc-6571-4e65-a50c-0bd4b8d63b74
function poly_expand(x; order = 2) # expand the design matrix to the pth order
	n = length(x)
	return hcat([x.^p for p in 0:order]...)
end;

# ╔═╡ 4154585d-4eff-4ee9-8f33-dad0dfdd143c
function poly_reg(x, y; order = 2) # fit a polynomial regression to the input x; x is assumed a vector
	X = poly_expand(x; order=order)
	w = linear_reg_normal_eq(X, y)
	l = loss(w, X, y)
	return w, l
end;

# ╔═╡ 46180264-bddc-47d8-90a7-a16d0ea87cfe
poly_fun(x, w) = sum([x^p for p in 0:length(w)-1] .* w);

# ╔═╡ 9aff2235-7da4-41a9-9926-4fe291d9a638
let
	gr()
	w, l = poly_reg(x_room, df_house.target; order=poly_order)
	@df df_house scatter(:rm, :target, xlabel="room", ylabel="price", label="", title="Poly-regression order: "* L"%$(poly_order)" *"; training loss = "*L"%$(round(l/length(df_house.target); digits=2))")
	plot!(3:0.1:9, (x) -> poly_fun(x, w), lw=2, label=L"h(\mathbf{x})", legend=:outerbottom, ylim=[-5, 65])
end

# ╔═╡ 86963d09-8235-45c3-a3d7-60f1ba5f17f4
md"""


# Fixed basis expansion
"""

# ╔═╡ a59c0d8f-1f6a-4576-b73a-b2314c254d26
md"""

## Fixed basis expansion

"""

# ╔═╡ 0335b7f0-cbd1-4688-816e-5f9b057c87ce
md"""

More **generally**, **fixed basis expansion** method expands via a set of ``K`` **basis functions**

```math
\large
\Phi=\;\;\;\;\;\;\{\textcolor{lightblue}{\phi_0(\mathbf{x})},
\;\;\;\;\;\;\;\textcolor{orange}{\phi_1(\mathbf{x})},\;\;\;\;\;\; \textcolor{green}{\phi_2(\mathbf{x})},\;\;\;\; \ldots,\;\;\;\textcolor{magenta}{\phi_K(\mathbf{x})}\}
```

The **expanded** input matrix is a ``n \times K+1`` matrix: ``\mathbf{X} \rightarrow_{{\Phi}} \boldsymbol{\Phi}``

```math
\large
\boldsymbol{\Phi} = \begin{bmatrix}\columncolor{lightblue}\phi_0(\mathbf{x}^{(1)}) & \columncolor{orange}\phi_1(\mathbf{x}^{(1)}) & \columncolor{lightgreen}\phi_2(\mathbf{x}^{(1)})  & \ldots & \columncolor{magenta}\phi_p(\mathbf{x}^{(1)}) \\
\phi_0(\mathbf{x}^{(2)}) & \phi_1(\mathbf{x}^{(2)}) & \phi_2(\mathbf{x}^{(2)}) & \ldots & \phi_p(\mathbf{x}^{(2)})\\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\phi_0(\mathbf{x}^{(n)}) & \phi_1(\mathbf{x}^{(n)}) & \phi_2(\mathbf{x}^{(n)}) & \ldots & \phi_p(\mathbf{x}^{(n)})
\end{bmatrix}
```
"""

# ╔═╡ e46303c4-fc28-4a2c-8f6a-11ca84188553
md"""

## For example


**Polynomial regression** is a specific case, where


```math
\large
\colorbox{lightblue}{$\phi_0(x) = 1$}
```


```math
\large
\boldsymbol{\Phi}  = \begin{bmatrix}\columncolor{lightblue}1 & \;\cdot\; & \;\;\;\cdot\;\;\;  & \ldots & \;\;\;\cdot\;\;\; \\
1 & \;\;\;\cdot\;\;\; & \;\;\;\cdot\;\;\; & \ldots & \;\;\;\cdot\;\;\;\\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \;\cdot\; & \;\;\;\cdot\;\;\; & \ldots & \;\;\;\cdot\;\;\;
\end{bmatrix}
```

"""

# ╔═╡ a428e3d2-090e-4444-8486-b98137baf2c6
md"""

## For example


**Polynomial regression** is a specific case, where


```math
\large
\colorbox{lightblue}{$\phi_0(x) = 1$}\, \colorbox{orange}{$\phi_1(x) =x $}
```


```math
\large
\boldsymbol{\Phi}  = \begin{bmatrix}\columncolor{lightblue}1 & \columncolor{orange}{x}^{(1)} & \;\;\cdot\;\;  & \ldots & \;\;\cdot\;\;\\
1 & {x}^{(2)} & \;\;\cdot\;\; & \ldots & \;\;\cdot\;\;\\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & {x}^{(n)} & \;\;\cdot\;\; & \ldots &\;\;\cdot\;\;
\end{bmatrix}
```

"""

# ╔═╡ cb07a067-d437-47bc-b2b5-73dd07755a8e
md"""

## For example


**Polynomial regression** is a specific case, where


```math
\large
\colorbox{lightblue}{$\phi_0(x) = 1$}\, \colorbox{orange}{$\phi_1(x) =x $}\, \colorbox{lightgreen}{$\phi_2(x) = x^2 $}
```

```math
\large
\boldsymbol{\Phi}  = \begin{bmatrix}\columncolor{lightblue}1 & \columncolor{orange}{x}^{(1)} & \columncolor{lightgreen}({x}^{(1)})^2  & \ldots & \;\;\cdot\;\;\\
1 & {x}^{(2)} & ({x}^{(2)})^2 & \ldots & \;\;\cdot\;\;\\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & {x}^{(n)} & ({x}^{(n)})^2 & \ldots &\;\;\cdot\;\;
\end{bmatrix}
```

"""

# ╔═╡ a3f19da3-a51c-4997-87e1-e3976c220b27
md"""

## For example


**Polynomial regression** is a specific case, where

```math
\large
\colorbox{lightblue}{$\phi_0(x) = 1$}\, \colorbox{orange}{$\phi_1(x) =x $}\, \colorbox{lightgreen}{$\phi_2(x) = x^2 $}\, \ldots\, \colorbox{magenta}{$\phi_p(x) = x^p$}
```

```math
\large
\tilde{\mathbf{X}} = \begin{bmatrix}\columncolor{lightblue}1 & \columncolor{orange}{x}^{(1)} & \columncolor{lightgreen}({x}^{(1)})^2  & \ldots & \columncolor{magenta} ({x}^{(1)})^p\\
1 & {x}^{(2)} & ({x}^{(2)})^2 & \ldots & ({x}^{(2)})^p\\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & {x}^{(n)} & ({x}^{(n)})^2 & \ldots &({x}^{(n)})^p\;
\end{bmatrix}
```



or *i.e.* simply the basis functions are 

$\boxed{\Large \phi_k(x) = x^k}$
"""

# ╔═╡ 0c2c15cb-8bbb-496c-b561-033197c9da5e
md"""
## Polynomial basis functions



The polynormial basis functions 

$\boxed{\Large \phi_k(x) = x^k, \;\; \text{for }k = 0,1,\ldots, K}$

are **NOT** **location** based

* it only depends on the power ``k``
* does not depend on *e.g.* ``x``'s range
"""

# ╔═╡ 78daf3d9-8070-43ea-a9a1-970e317c3006
let
		plt = plot(title="Polynomial basis functions", legend=:outerright)
		for k in 1:10
			plot!(-1:0.01:1, (x) -> x^k, label=L"\phi_{%$(k)} = x^{%$(k)}", lw=2)
		end
	plt
end

# ╔═╡ 8c239067-7046-4eaf-9b68-b92f98bf9aeb
md"""

## Other basis function -- radian basis functions


More commonly used basis functions are **location-aware**

* ``\phi`` depends on an extra expansion location parameter ``\mu``

**For example**, the **radian-basis-function (rbf)**


```math
\large
\phi(x; \mu)=\text{rbf}(x;\mu) = \exp\left\{ -\frac{1}{2s} (x-\mu)^2\right\}
```
* ``\mu``: the expansion location
* ``s>0``: scale parameter


"""

# ╔═╡ 51efd68e-175d-4008-bbba-0e6d4d32d1b4
aside(tip(md"For multi-dimensional input, 


```math
\phi(\mathbf{x}; \boldsymbol{\mu}) = \exp\left\{ -\frac{1}{2s} (\mathbf{x}-\boldsymbol{\mu})^\top(\mathbf{x}-\boldsymbol{\mu})\right\}

```
"))

# ╔═╡ ca328eb5-c51f-4d95-91ad-685741eaeb2e
md"Select ``\mu``: $(@bind μ_ Slider(-1.5:0.2:1.5; show_value=true)),
Select ``s``: $(@bind s_ Slider(0.01:0.01:1; default=.25, show_value=true))"

# ╔═╡ f795f280-26c6-4356-b886-98d577cfeac8
let
	plt_rbf_basis = plot(title="RBF basis functions", framestyle=:origin)
	s = s_
	for μ in -1.5:0.2:1.5
		plot!(-2:0.01:2, (x) -> exp(-(x-μ)^2/(2*s^2)), lw=1, lc=:gray, label="")
	end

	plot!(-2:0.01:2, (x) -> exp(-(x-μ_)^2/(2*s^2)), lw=4, lc=2, label=L"\phi(x; \mu = %$(μ_))", legend=:outerright)
	plt_rbf_basis
end

# ╔═╡ f5636856-dee0-4de6-b0c0-7c98bdc83524
md"""

## Other basis function -- logistic


More commonly used basis functions are **location-aware**

* ``\phi`` depends on an extra expansion location parameter ``\mu``

**For example**, the **logistic** function (aka sigmoid)



```math
\phi(x; \mu) = \frac{1}{1+ e^{-\frac{(x-\mu)}{s}}}
```

* a squash function: squeeze to ``(0, 1)``

"""

# ╔═╡ 23b9ee29-f5c5-485c-8d3e-870b5407a5ae
# aside(tip(md"For multi-dimensional input, 


# ```math
# \phi(\mathbf{x}; \boldsymbol{\mu}, \mu_0) = \frac{1}{1+ e^{-(\boldsymbol{\mu}^\top\mathbf{x}+\mu_0)}}


# ```
# "))

# ╔═╡ 49d1585d-9c37-43bb-80d5-a854290534a5
md"Select ``\mu``: $(@bind μl_ Slider(-1.5:0.01:1.5; default=0, show_value=true)),
Select ``s``: $(@bind sl_ Slider(0.01:0.01:1; default=.25, show_value=true))"

# ╔═╡ a9f1428c-2ea0-4903-b1e2-18f946252451
let
	plt_sigmoid_basis = plot(title="Logistic basis functions", framestyle=:origin)
	s_sig = sl_
	for μ in -1:0.1:1
		plot!(-1.5:0.01:1.5, (x) -> logistic((x-μ)/s_sig),ylim=[-0.3, 1.3], lw =1,lc=:gray, label="")
	end

	plot!(-1.5:0.01:1.5, (x) -> logistic((x -μl_)/s_sig), lw =3,lc=:red, label=L"\phi(x; \mu = %$(μl_))", legend=:outerbottom)

	plot!(-1.5:0.1:1.5, (x) -> 1, lw=2, ls=:dash,lc=:black, label="")
	plot!(-1.5:0.1:1.5, (x) -> 0, lw=2, ls=:dash,lc=:black, label="")
	plt_sigmoid_basis
end

# ╔═╡ db1b3894-a92d-4b38-bb7d-4ab5c5469549
md"""

## Other basis function -- ReLu


More commonly used basis functions are **location-aware**

* ``\phi`` depends on an extra expansion location parameter ``\mu``

**For example**, the **re**ctified **l**inear **u**nit  (or **ReLu** function )


```math
\phi(x; \mu)=\text{Relu}(x;\mu) = \begin{cases} 
x-\mu & x > \mu\\
0 & x \leq \mu
\end{cases}
```




"""

# ╔═╡ dc801bff-89c8-49b7-8b3c-123f240f4184
md"Select ``\mu``: $(@bind μr_ Slider(-5:0.05:5; default=0, show_value=true)),
add other $(@bind add_others_relu CheckBox(default=false))
"

# ╔═╡ 254fb3d0-a1ac-43cb-bc12-3d512aa6184e
let
	plt_relu_basis = plot(title="ReLu basis functions")
	plot!(-6:0.05:6, (x) -> max(0,x-μr_), lw=2,label="", lc=:red, framestyle=:origin)

	if add_others_relu
		for μ in -5:1:5
			plot!(-6:0.05:6, (x) -> max(0,x-μ), lw=1,  label="", lc=:gray)
		end
	end

	plt_relu_basis
end

# ╔═╡ c83c7e50-532c-4fb2-8c6d-5f3af32d205e
md"""

## Summary
"""

# ╔═╡ 80cdaf0c-c863-49db-86df-71413068e5f4
begin
	plt_poly_basis = plot(title="Polynomial basis functions")
	for k in 1:10
		plot!(-1:0.05:1, (x) -> x^k, label="")
	end


	plt_rbf_basis = plot(title="RBF basis functions")
	s = 0.25
	for μ in -1:0.2:1
		plot!(-1:0.01:1, (x) -> exp(-(x-μ)^2/(2*s^2)), label="")
	end



	plt_sigmoid_basis = plot(title="Logistic basis functions")
	s_sig = 0.1
	for μ in -1:0.2:1
		plot!(-1:0.01:1, (x) -> logistic((x-μ)/s_sig), label="")
	end


	plt_relu_basis = plot(title="ReLu basis functions")
	for μ in -1:0.2:1
		plot!(-1:0.01:1, (x) -> max(0,x-μ), label="")
	end
	plot(plt_poly_basis, plt_rbf_basis, plt_sigmoid_basis, plt_relu_basis)
end

# ╔═╡ 2f19637e-80a2-4855-a7b7-2d70b25e188c
md"""

## Demonstration -- sinusoid function

"""

# ╔═╡ bc513037-c689-4eca-865d-d94a6a8aa997
begin
	Random.seed!(123)
	true_f(x) = sin(x)
	nobs_ = 25
	xs_q4 = collect(range(0, 2π, nobs_))
	ys_q4 = true_f.(xs_q4) + randn(nobs_)/4

end;

# ╔═╡ 88c1bb3e-dab8-459a-b2d0-ebc2496f5b19
md"Add true function: $(@bind addsin CheckBox(default=false))"

# ╔═╡ 3bfb67fe-0d00-4bcd-80f0-aee3b9bd6bc4
let
	plt = plot(xs_q4, ys_q4, st=:scatter, label="training data", title="Regression training data")
	if addsin
		plot!(0:0.1:2π, true_f, label="true signal", lw=2, title=L"\sin(x)")
	end
	plt
end

# ╔═╡ d892fc55-3d19-4c12-a275-ba94e2fbaba0
md"""

## Fixed basis expansion regression



"""

# ╔═╡ 718f99ce-640c-47c4-bc91-9b650eb88a36
md"Number of basis: $(@bind n_basis Slider(1:1:15; default=1, show_value=true))"

# ╔═╡ 1557a74b-be97-461e-b9d7-d31f2728120f
md"Add fixed basis regression: $(@bind add_fixed_fit CheckBox(default=false))"

# ╔═╡ 64984c2c-bc17-41c3-a543-19b4d7b3bdb2
begin
	rbf(s=1) = (z) -> exp(-.5 * z^2/s)
	relu(s=1) = (z) -> max(z, 0)
	sigmoid(s=1) = (z) -> logistic(z/s)
end;

# ╔═╡ e9bfad75-eb43-4212-a141-0e0655c48758
md"Basis function: $(@bind ϕ Select([rbf, relu, sigmoid])), 
Scale parameter ``s`` (if applied): $(@bind scale_ Slider(0.01:0.01:3.0;default=1.0, show_value=true))
"

# ╔═╡ a680773b-b385-4938-ada5-348299a29f02
function basis_expansion(xs, μs, σ::Function; intercept=true)
	n = length(xs)
	p = length(μs) 
	Φ = zeros(n, p)
	for (j, μ) in enumerate(μs)
		Φ[:, j] = σ.(xs .- μ)
	end
	if intercept 
		return [ones(n) Φ]
	else
		Φ
	end
end;

# ╔═╡ f1c90208-4222-49d3-9f07-c19fec64cef4
begin
	Random.seed!(111)
	n_fixed_basis = n_basis
	μs = range((extrema(xs_q4) .+ (-.5, .5))..., n_fixed_basis+2)[2:end-1]
end;

# ╔═╡ ce4f4d96-3f01-46d2-ba1e-6108a7af9141
linear_reg(Φ, y; λ = 1e-10) =  Φ \ y;

# ╔═╡ 852d5317-3f62-4ac8-9b8a-ab0b7869b269
let
	basis_fun = ϕ(scale_)
	plt = plot(xs_q4, ys_q4, st=:scatter, label="training data", ylim=[-1.25, 1.25], framestyle=:origin)
	for (j, μ) in enumerate(μs)
		if j == 1
			plot!(-.5:0.1:2π+.5, (x) -> basis_fun(x-μ), lw=1, ls=:dash, lc=:gray, label="basis functions")
		else
			plot!(-.5:0.1:2π+.5, (x) -> basis_fun(x-μ), lw=1, ls=:dash, lc=:gray, label="")
		end
	end
	
	plot!(-.5:0.1:(2π+.5), true_f, label="true signal", lc =:blue, lw=1.5, title="Fixed basis expansion regression dataset", legend=:bottomleft)

	if add_fixed_fit
		Φ = basis_expansion(xs_q4, μs, basis_fun; intercept=false)
		w = linear_reg(Φ, ys_q4; λ = 0.0)
		plot!(-0.5:0.1:2π+.5, (x) -> (basis_expansion([x], μs, basis_fun;  intercept=false) * w)[1] , label="fixed basis regression", lw=3, lc=2, ylim=[-1.25, 1.25])
	end


	plt
end

# ╔═╡ 51d84692-edfa-4226-93ce-50f999a347cb
md"""

## Summary


All regressions are fit with ``K=8`` basis expansions

"""

# ╔═╡ 87ccd668-2350-4cd9-aa11-8386a3aecc4d
let
	n_basis = 8
	μs = range((extrema(xs_q4) .+ (-.5, .5))..., n_basis+2)[2:end-1]
	plt_rbf = plot(title="Rbf basis regression")
	basis_fun = rbf(1.0)
	plot!(xs_q4, ys_q4, st=:scatter, label="", ylim=[-1.25, 1.25], framestyle=:origin)
	Φ = basis_expansion(xs_q4, μs, basis_fun)
	w = linear_reg(Φ, ys_q4; λ = 0.0)
	plot!(-0.5:0.1:2π+.5, (x) -> (basis_expansion([x], μs, basis_fun) * w)[1] , label="", lw=3, lc=2, ylim=[-1.25, 1.25])




	plt_sigmoid = plot(title="Sigmoid basis regression")
	basis_fun = sigmoid(1.0)
	plot!(xs_q4, ys_q4, st=:scatter, label="training data", ylim=[-1.25, 1.25], framestyle=:origin)
	Φ = basis_expansion(xs_q4, μs, basis_fun)
	w = linear_reg(Φ, ys_q4; λ = 0.0)
	plot!(-0.5:0.1:2π+.5, (x) -> (basis_expansion([x], μs, basis_fun) * w)[1] , label="", lw=3, lc=2, ylim=[-1.25, 1.25])



	plt_relu = plot(title="ReLu basis regression")
	basis_fun = relu(1.0)
	plot!(xs_q4, ys_q4, st=:scatter, label="training data", ylim=[-1.25, 1.25], framestyle=:origin)
	Φ = basis_expansion(xs_q4, μs, basis_fun)
	w = linear_reg(Φ, ys_q4; λ = 0.0)
	plot!(-0.5:0.1:2π+.5, (x) -> (basis_expansion([x], μs, basis_fun) * w)[1] , label="", lw=3, lc=2, ylim=[-1.25, 1.25])



	
	plt_poly = plot(title="Polynomial basis regression")
	w, l = poly_reg(xs_q4, ys_q4; order=n_basis)
	plot!(xs_q4, ys_q4, st=:scatter, label="training data", ylim=[-1.25, 1.25], framestyle=:origin)
	# Φ = basis_expansion(xs_q4, μs, basis_fun)
	# w = linear_reg(Φ, ys_q4; λ = 0.0)
	plot!(-0.5:0.1:2π+.5, (x) ->  poly_fun(x, w) , label="", lw=3, lc=2, ylim=[-1.25, 1.25])


	plot(plt_poly, plt_rbf, plt_sigmoid, plt_relu)

end

# ╔═╡ 9e8ef52f-a9bc-4754-bb6e-98979384da23
md"""

# Overfit & Underfit
"""

# ╔═╡ 882da1c1-1358-4a40-8f69-b4d7cbc9387e
md"""

## Another dataset


The true function is a 4-th power polynomial:

```math
h(x) = -10 x + 2 x^2 + 3 x^3 - x^4
```
"""

# ╔═╡ 0dff10ec-dd13-4cc2-b092-9e72454763cc
begin
	gr()
	Random.seed!(123)
	x_poly = [range(-1.8, -0.5, length=8)... range(.5, 1.8, length=8)...][:]
	x_poly_test = -2 : 0.1: 2
	w_poly = [0.0, -5, 5, 2, -2]
	y_poly = [poly_fun(x, w_poly) for x in x_poly] + randn(length(x_poly))
	y_poly_test = [poly_fun(x, w_poly) for x in x_poly_test] + randn(length(x_poly_test))
	plot(x_poly, y_poly, st=:scatter, label="training data", xlabel=L"x", ylabel=L"y")

	plot!(x_poly_test, y_poly_test, st=:scatter, lc=2, alpha= .5, label="testing data")
	plot!(-2:0.1:2, (x) -> poly_fun(x, w_poly), label="true "*L"h(x)", lw=2, lc=2, size=(600, 400))
end

# ╔═╡ 726a1008-26e6-417d-a73a-57e32cb224b6
md"""

## Underfitting 

Fit polynomial regression with orders: ``p = 1, 2, 4, 7, 10, 15``
"""

# ╔═╡ 8cc45465-78e7-4777-b0c7-bb842e4e51a8
let
	gr()
	poly_order = [1, 2, 3, 7, 10, 15]
	ylim = [extrema(y_poly)...;]
	ylim += [-1.0, 1.0]
	plots_ =[]
	for p in poly_order
		plt = plot(x_poly, y_poly, st=:scatter, label="")
		w, loss = poly_reg(x_poly, y_poly; order=p)
		plot!(-2:0.02:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=ylim, title="training loss: "*L"%$(round(loss; digits=2))")
		push!(plots_, plt)
	end
	plot(plots_..., size=(900, 600))
end

# ╔═╡ fc2206f4-fd0f-44f8-ae94-9dae77022fab
md"""

!!! note "Underfitting"
	Lower-order models lead to overly simplified models
    * the performance is not adequate

!!! note "Overfitting"
	Higher-order models lead to overly complicated models
    * checking training errors only does not help
    * training error favours complicated models
"""

# ╔═╡ c91fc381-613c-4c9b-98b6-f4dd897a69a5
md"""
## Testing performance


Now check the performance on the reserved test data

For ``({x}^i, y^i) \in \mathcal{D}_{test}``:
```math
L_{test}(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^n (y^{(i)} - h(\mathbf{x}^{(i)}; \hat{\mathbf{w}}))^2
```
where ``\hat{\mathbf{w}}`` is estimated by the least squared method
"""

# ╔═╡ 21b6b547-66ed-4231-830b-1a09adaf400c
let
	gr()
	poly_order = [1, 2, 4, 7, 10, 15]


	plots_ =[]
	for p in poly_order
		plt = plot(x_poly, y_poly, st=:scatter, ms=3, mc=1,alpha=0.5, label="train data")
		plot!(x_poly_test, y_poly_test, st=:scatter, ms=3, mc=2,alpha=0.5,  label="test data")
		w, loss = poly_reg(x_poly, y_poly; order=p)
		loss_test = norm([poly_fun(x, w) for x in x_poly_test] - y_poly_test)/length(y_poly_test)
		plot!(-1.5:0.01:1.5, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=[-1, 7], title="test loss: "*L"%$(round(loss_test; digits=2))")
		push!(plots_, plt)
	end
	plot(plots_..., size=(900, 600))
end

# ╔═╡ ce8cd98f-da70-476c-817c-026365b982c0
md"""

## Regularisation



One technique to avoid overfitting is **regularisation**

* add a penalty term to the normal loss function


```math
L(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^n (y^{(i)} - h(\mathbf{x}^{(i)}; \mathbf{w}))^2 + \frac{\lambda}{2} \sum_{j=1}^{m} w_j^2
```
* ``\lambda \geq 0`` is a hyperparameter


How it works ?
* large ``w_j`` is penalised
* large ``\|\mathbf{w}\| = \sqrt{ \mathbf{w}^\top \mathbf{w}}`` implies very wiggly prediction function

Note that ``\mathbf{w}^\top \mathbf{w} = \sum_j w_j^2``

The regularised loss for linear regression can be written in matrix notation

```math
L(\mathbf{w}) = \frac{1}{2}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + \frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}
```

This is called **ridge regression**.
"""

# ╔═╡ 8436c5ce-d7be-4c02-9e89-7da701303263
aside(tip(md"""

Recall 

```math
\nabla_{\mathbf{w}} \mathbf{w}^\top\mathbf{w} = 2 \mathbf{w}
```

"""))

# ╔═╡ 6a9e0bd9-8a2a-40f0-aaac-bf32d39ffac8
md"""
## Ridge regression

The problem now is to optimise the regularised loss

```math

\mathbf{w}_{ridge} \leftarrow \arg\min_{\mathbf{w}} \frac{1}{2}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + \frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}
```

Its gradient is 


```math
\nabla L(\mathbf{w}) = \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) + \lambda\mathbf{w}
```
"""

# ╔═╡ df62e796-d711-4ea1-a0c2-e3ec6c501a78
md"""

Set the gradient to zero, and solve it


!!! hint "Hint"
	```math
		\lambda \mathbf{w} = \lambda \mathbf{Iw},
	```
	where ``\mathbf{I}`` is the identity matrix.

!!! note "Ridge regression solution"
	```math
		\mathbf{w}_{ridge} = (\mathbf{X}^\top\mathbf{X} +\lambda \mathbf{I})^{-1} \mathbf{X}^\top\mathbf{y}
	```

"""

# ╔═╡ 8493d307-9158-4821-bf3b-c368d9cd5fc5
ridge_reg(X, y; λ = 1) = (X' * X + λ *I)^(-1) * X' * y

# ╔═╡ 88d98d87-f3cf-42f4-9282-1e6f383934cd
md"""

## Effect of hyperparameter


Polynomial ridge regressions (polynomial order is ``10``) with different hyperparameters

```math
\lambda = [ 0, 1, 5, 10, 20, e^{20}]
```


```math
e^{20} \approx 4.8 \times 10^8
```

"""

# ╔═╡ 0e60180d-c4eb-4c83-9301-e3151ab828d5
let
	gr()
	poly_order = 10
	λs = [0, 1, 5, 10, 20, exp(20)]
	plots_ =[]
	for λ in λs
		plt = plot(x_poly, y_poly, st=:scatter, ms=3, mc=1,alpha=0.5, label="train data")
		plot!(x_poly_test, y_poly_test, st=:scatter, ms=3, mc=2,alpha=0.5,  label="test data")
		x_p = poly_expand(x_poly; order = poly_order)
		w = ridge_reg(x_p, y_poly; λ = λ)
		loss_test = norm([poly_fun(x, w) for x in x_poly_test] - y_poly_test)/length(y_poly_test)
		plot!(-2:0.05:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="", legend=:outerbottom, ylim=[-1, 7], title="test loss: "*L"%$(round(loss_test; digits=2));\;" * L"\lambda=%$(round(λ;digits=1))")
		push!(plots_, plt)
	end
	plot(plots_..., size=(900, 600))
end

# ╔═╡ 58d6a11e-a375-4f0b-84d7-b95e7cfd3033
md"""

## Question 



!!! question "Question"
	What is the optimisation result?
	```math
	\mathbf{w} \leftarrow \arg\min_{\mathbf{w}} \frac{1}{2}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + \frac{e^{20}}{2} (\mathbf{w} - \mathbf{1})^\top (\mathbf{w} - \mathbf{1})
	```

"""

# ╔═╡ 88ef37e0-d6e9-4cb6-b765-31b5270a5f89
md"""

## How to set hyperparameter ``\lambda``



!!! question "Question"
	Can we optimise ``\lambda`` (note that ``\lambda \geq 0``) by minimising the loss with training data? *i.e.*

	```math
	\lambda \leftarrow \arg\min_{\lambda} \frac{1}{2}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + \frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}
	```

"""

# ╔═╡ 979d7298-b58e-40ed-84d2-580118f1c41d
md"""

## How to set hyperparameter ``\lambda`` -- cross-validation


Cross-validation (CV) is an empirical method 
* to compare models in general
* as well as setting hyperparameters (also a form of model comparison)


``K``--fold cross-validation: (``K=5``)
* split the data to ``K`` folds

"""

# ╔═╡ b38627c8-01be-4b2b-a4ff-590de8e9d337
html"<center><img src='https://scikit-learn.org/stable/_images/grid_search_cross_validation.png' width = '500' /></center>"

# ╔═╡ af4be079-8461-4c62-9e23-222c0dd1a738
md"""

* a model is trained using  ``K-1`` of the folds as training data;
* the resulting model is validated on the remaining part of the data as test data


Therefore, each model is fitted ``K`` times, and the aggregated performance is used to compare the models
* very computationally intensive!
* discretisation method for continuous ``\lambda``: not ideal

## Why CV?

Why fit ``K`` times rather than just one?
* repeated measurements give us more reliable performance estimation
  * we not only have a mean estimator but also error bars
* cross-validation is like a repeated controlled experiment 
  * the effect of random sampling error (of the training-testing data) is minimised
"""

# ╔═╡ fac1b011-d1bd-4c0f-9d82-a8e46d68b8c4
md"""

## Exercise


!!! exercise "Exercise"
	Implement a ``K=5`` CV method to find the optimal ``\lambda`` for the ridge regression
	* choose ``\lambda \in [0,1,2,3,\ldots, 20]``

"""

# ╔═╡ 974f1b58-3ec6-447a-95f2-6bbeda43f12f
md"""

# Appendix
"""

# ╔═╡ 238e7b56-fb3a-4e9b-9c31-09c1f4a1df2a
begin
	Random.seed!(111)
	num_features = 2
	num_data = 25
	true_w = rand(num_features+1) * 10
	# simulate the design matrix or input features
	X_train = [ones(num_data) rand(num_data, num_features)]
	# generate the noisy observations
	y_train = X_train * true_w + randn(num_data)
end;

# ╔═╡ cb02aee5-d082-40a5-b799-db6b4af557f7
# md"""
# ## More datasets


# It turns out linear correlations are more common than we expect!

# * *e.g.* the flipper size and body mass of Penguins 
# """

# ╔═╡ 8deb1b8c-b67f-4d07-8986-2333dbadcccc
# md"""
# ![](https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png)"""

# ╔═╡ af622189-e504-4633-9d9e-ab16c7293f82
df_penguin = let
	Logging.disable_logging(Logging.Warn)
	df_penguin = DataFrame(CSV.File(download("https://gist.githubusercontent.com/slopp/ce3b90b9168f2f921784de84fa445651/raw/4ecf3041f0ed4913e7c230758733948bc561f434/penguins.csv"), types=[Int, String, String, [Float64 for _ in 1:4]..., String, Int]))
	df_penguin[completecases(df_penguin), :]
end;

# ╔═╡ 9267c4a4-74d1-4515-95d6-acc3b12e5ed6
# first(df_penguin, 5)

# ╔═╡ 76cc8ca7-d17e-4cd7-a6de-4d606e0a0985
# @df df_penguin scatter(:flipper_length_mm, :body_mass_g, group = (:species), legend=:topleft, xlabel="Flipper length", ylabel="Body mass");

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.10"
DataFrames = "~1.5.0"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.21"
LogExpFunctions = "~0.3.23"
MLDatasets = "~0.7.9"
Plots = "~1.38.14"
PlutoTeachingTools = "~0.2.11"
PlutoUI = "~0.7.51"
StatsBase = "~0.34.0"
StatsPlots = "~0.15.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "3c5c0fa779a72e256b388f8a09587c9ff0e6b889"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "2b301c2388067d655fe5e4ca6d4aa53b61f895b4"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.31"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

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

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "54b00d1b93791f8e19e31584bd30f2cb6004614b"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.38"

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

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BufferedStreams]]
git-tree-sha1 = "bb065b14d7f941b8617bc323063dbe79f55d16ea"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.1.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "ed28c86cbde3dc3f53cf76643c2e9bc11d56acc7"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.10"

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

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.Chemfiles]]
deps = ["Chemfiles_jll", "DocStringExtensions"]
git-tree-sha1 = "6951fe6a535a07041122a3a6860a63a7a83e081e"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.40"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3743181e30d87c23d9c8ebd493b77f43d8f1890"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.4+0"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "a6e6ce44a1e0a781772fc795fb7343b1925e9898"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.2"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "d730914ef30a06732bdd9f763f6cc32e92ffbff1"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "bc0a264d3e7b3eeb0b6fc9f6481f970697f29805"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.10"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

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

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c72970914c8a21b36bbc244e9df0ed1834a0360b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.95"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

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

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f9818144ce7c8c41edf5c4c179c684d92aa4d9fe"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.6.0"

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

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "ed569cb9e7e3590d5ba884da7edc50216aac5811"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.1.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

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

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8b8a2fd4536ece6e554168c21860b6820a8a83db"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "19fad9cd9ae44847fe842558a744748084a722d1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.7+0"

[[deps.GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

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

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

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

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "c73fdc3d9da7700691848b78c61841274076932a"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.15"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e77dbf117412d4f164a464d610ee6050cc75272"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.6"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "84204eae2dd237500835990bcade263e27674a93"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.16"

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

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "ce28c68c900eed3cdbfa418be66ed053e54d4f56"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.7"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "6667aadd1cdee2c6cd068128b3d226ebc4fb0c67"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.9"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "42c17b18ced77ff0be65957a591d34f4ed57c631"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.31"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "SnoopPrecompile", "StructTypes", "UUIDs"]
git-tree-sha1 = "84b10656a41ef564c39d2d477d7236966d2b5683"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.12.0"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6a125e6a4cb391e0b9adbd1afa9e771c2179f8ef"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.23"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "47be64f040a7ece575c2b5f53ca6da7b548d69f4"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.4"

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
git-tree-sha1 = "26a31cdd9f1f4ea74f649a7bf249703c687a953d"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "5.1.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "09b7505cc0b1cee87e5d4a26eea61d2e1b0dcd35"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.21+0"

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
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "6eff5740c8ab02c90065719579c7aa0eb40c9f69"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.4"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLDatasets]]
deps = ["CSV", "Chemfiles", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Printf", "Requires", "SparseArrays", "Tables"]
git-tree-sha1 = "498b37aa3ebb4407adea36df1b244fa4e397de5e"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.9"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "FoldsThreads", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "ca31739905ddb08c59758726e22b9e25d0d1521b"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

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
git-tree-sha1 = "99e6dbb50d8a96702dc60954569e9fe7291cc55d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.20"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

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

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a5aef8d4a6e8d81f171b2bd4be5265b01384c74c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.10"

[[deps.Pickle]]
deps = ["DataStructures", "InternedStrings", "Serialization", "SparseArrays", "Strided", "StringEncodings", "ZipFile"]
git-tree-sha1 = "e6a34eb1dc0c498f0774bbfbbbeff2de101f4235"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

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

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ad59edfb711a4751e0b8271454df47f84a47a29e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.14"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

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
git-tree-sha1 = "88222661708df26242d0bfb9237d023557d11718"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.11"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

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
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "213579618ec1f42dea7dd637a42785a608b1ea9c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

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

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "6d7bb727e76147ba18eed998700998e17b8e4911"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.4"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

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
git-tree-sha1 = "feafdc70b2e6684314e188d95fe66d116de834a7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.2"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

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

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
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

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "8982b3607a212b070a5e46eea83eb62b4744ae12"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.25"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

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
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "14ef622cf28b05e38f8af1de57bc9142b03fbfe3"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.5"

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "a7a664c91104329c88222aa20264e1a05b6ad138"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.3"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "33c0da881af3248dafefb939a21694b97cfece76"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.6"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

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

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "25358a5f2384c490e98abd565ed321ffae2cbb37"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.76"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "ba4aa36b2d5c98d6ed1f149da916b3ba46527b2b"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.14.0"
weakdeps = ["InverseFunctions"]

    [deps.Unitful.extensions]
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "ea37e6066bf194ab78f4e747f5245261f17a7175"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.2"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

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

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

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
# ╟─17a3ac47-56dd-4901-bb77-90171eebc8c4
# ╟─29998665-0c8d-4ba4-8232-19bd0de71477
# ╟─cb72ebe2-cea8-4467-a211-5c3ac7af74a4
# ╟─358dc59c-8d06-4272-9a13-6886cdaf3dd9
# ╟─9bd2e7d6-c9fb-4a67-96ef-049f713f4d53
# ╟─cf9c3937-3d23-4d47-b329-9ecbe0006a1e
# ╟─dfcfd2c0-9f51-48fb-b91e-629b6934dc0f
# ╟─c4e497fc-cfbf-4d0b-9a0c-1071f2f43a98
# ╟─0e2dc755-57df-4d9a-b4f3-d01569c3fcde
# ╟─c4f42980-0e68-4943-abbe-b28f05dd3ee5
# ╟─8b479e5c-8151-4769-9d27-b5661285b497
# ╟─55fbb81b-8323-4405-beb9-acd557e6d9f3
# ╟─86f09ee8-087e-47ac-a81e-6f8c38566774
# ╟─7ab7721d-90e7-4606-8614-f0f5470c9d29
# ╟─f8f34671-ef4c-4300-a983-10d42d43fb9f
# ╟─8fbcf6c3-320c-47ae-b4d3-d710a120eb1a
# ╟─f65644e7-cb25-46ad-b146-87cf7de69f72
# ╟─22cbb1aa-c53f-45d6-891a-90c6f2b9e886
# ╟─5148115c-4b3d-4845-97fa-02b68d09ddaa
# ╟─ec41e396-5d30-4b07-9520-6f7a9c5da73b
# ╟─dd162f70-73b7-4f1a-beab-6fa26b2b11b1
# ╟─a495ee86-e78b-46e5-9be4-c4bbfad165f8
# ╟─d37f5e8f-63a6-440e-b6c0-ae283ded1eb9
# ╟─ce35ddcb-5018-4cb9-b0c9-01fb4b14be40
# ╟─2cdd8751-7ec8-47b0-a174-fdc23e176921
# ╟─720774c4-9aec-4329-bc90-51350fea0191
# ╟─8b596e98-4d0c-471e-bb25-20f492a9199b
# ╟─9aff2235-7da4-41a9-9926-4fe291d9a638
# ╟─edc245bc-6571-4e65-a50c-0bd4b8d63b74
# ╟─4154585d-4eff-4ee9-8f33-dad0dfdd143c
# ╟─46180264-bddc-47d8-90a7-a16d0ea87cfe
# ╟─86963d09-8235-45c3-a3d7-60f1ba5f17f4
# ╟─a59c0d8f-1f6a-4576-b73a-b2314c254d26
# ╟─0335b7f0-cbd1-4688-816e-5f9b057c87ce
# ╟─e46303c4-fc28-4a2c-8f6a-11ca84188553
# ╟─a428e3d2-090e-4444-8486-b98137baf2c6
# ╟─cb07a067-d437-47bc-b2b5-73dd07755a8e
# ╟─a3f19da3-a51c-4997-87e1-e3976c220b27
# ╟─0c2c15cb-8bbb-496c-b561-033197c9da5e
# ╟─78daf3d9-8070-43ea-a9a1-970e317c3006
# ╟─8c239067-7046-4eaf-9b68-b92f98bf9aeb
# ╟─51efd68e-175d-4008-bbba-0e6d4d32d1b4
# ╟─ca328eb5-c51f-4d95-91ad-685741eaeb2e
# ╟─f795f280-26c6-4356-b886-98d577cfeac8
# ╟─f5636856-dee0-4de6-b0c0-7c98bdc83524
# ╟─23b9ee29-f5c5-485c-8d3e-870b5407a5ae
# ╟─49d1585d-9c37-43bb-80d5-a854290534a5
# ╟─a9f1428c-2ea0-4903-b1e2-18f946252451
# ╟─db1b3894-a92d-4b38-bb7d-4ab5c5469549
# ╟─dc801bff-89c8-49b7-8b3c-123f240f4184
# ╟─254fb3d0-a1ac-43cb-bc12-3d512aa6184e
# ╟─c83c7e50-532c-4fb2-8c6d-5f3af32d205e
# ╟─80cdaf0c-c863-49db-86df-71413068e5f4
# ╟─2f19637e-80a2-4855-a7b7-2d70b25e188c
# ╟─bc513037-c689-4eca-865d-d94a6a8aa997
# ╟─88c1bb3e-dab8-459a-b2d0-ebc2496f5b19
# ╟─3bfb67fe-0d00-4bcd-80f0-aee3b9bd6bc4
# ╟─d892fc55-3d19-4c12-a275-ba94e2fbaba0
# ╟─718f99ce-640c-47c4-bc91-9b650eb88a36
# ╟─e9bfad75-eb43-4212-a141-0e0655c48758
# ╟─1557a74b-be97-461e-b9d7-d31f2728120f
# ╟─852d5317-3f62-4ac8-9b8a-ab0b7869b269
# ╟─64984c2c-bc17-41c3-a543-19b4d7b3bdb2
# ╟─a680773b-b385-4938-ada5-348299a29f02
# ╟─f1c90208-4222-49d3-9f07-c19fec64cef4
# ╟─ce4f4d96-3f01-46d2-ba1e-6108a7af9141
# ╟─51d84692-edfa-4226-93ce-50f999a347cb
# ╟─87ccd668-2350-4cd9-aa11-8386a3aecc4d
# ╟─9e8ef52f-a9bc-4754-bb6e-98979384da23
# ╟─882da1c1-1358-4a40-8f69-b4d7cbc9387e
# ╟─0dff10ec-dd13-4cc2-b092-9e72454763cc
# ╟─726a1008-26e6-417d-a73a-57e32cb224b6
# ╟─8cc45465-78e7-4777-b0c7-bb842e4e51a8
# ╟─fc2206f4-fd0f-44f8-ae94-9dae77022fab
# ╟─c91fc381-613c-4c9b-98b6-f4dd897a69a5
# ╟─21b6b547-66ed-4231-830b-1a09adaf400c
# ╟─ce8cd98f-da70-476c-817c-026365b982c0
# ╟─8436c5ce-d7be-4c02-9e89-7da701303263
# ╟─6a9e0bd9-8a2a-40f0-aaac-bf32d39ffac8
# ╟─df62e796-d711-4ea1-a0c2-e3ec6c501a78
# ╠═8493d307-9158-4821-bf3b-c368d9cd5fc5
# ╟─88d98d87-f3cf-42f4-9282-1e6f383934cd
# ╟─0e60180d-c4eb-4c83-9301-e3151ab828d5
# ╟─58d6a11e-a375-4f0b-84d7-b95e7cfd3033
# ╟─88ef37e0-d6e9-4cb6-b765-31b5270a5f89
# ╟─979d7298-b58e-40ed-84d2-580118f1c41d
# ╟─b38627c8-01be-4b2b-a4ff-590de8e9d337
# ╟─af4be079-8461-4c62-9e23-222c0dd1a738
# ╟─fac1b011-d1bd-4c0f-9d82-a8e46d68b8c4
# ╟─974f1b58-3ec6-447a-95f2-6bbeda43f12f
# ╟─238e7b56-fb3a-4e9b-9c31-09c1f4a1df2a
# ╟─cb02aee5-d082-40a5-b799-db6b4af557f7
# ╟─8deb1b8c-b67f-4d07-8986-2333dbadcccc
# ╟─f79bd8ab-894e-4e7b-84eb-cf840baa08e4
# ╟─af622189-e504-4633-9d9e-ab16c7293f82
# ╟─9267c4a4-74d1-4515-95d6-acc3b12e5ed6
# ╟─76cc8ca7-d17e-4cd7-a6de-4d606e0a0985
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002