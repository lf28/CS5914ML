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

# ╔═╡ 9f90a18b-114f-4039-9aaf-f52c77205a49
begin
	using LinearAlgebra
	using PlutoUI
	using PlutoTeachingTools
	using LaTeXStrings
	using Latexify
	using Random
	using Statistics
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	
end

# ╔═╡ 959d3f6e-ad5b-444f-9000-825063598837
using Zygote

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ 7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
ChooseDisplayMode()

# ╔═╡ bc96a33d-9011-41ec-a19e-d472cbaafb70
md"""

# CS5914 Machine Learning Algorithms


#### Vector calculus 2

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 653ab1b9-0839-4217-8301-f326efa2a2ad
md"""


# Multivariate calculus

"""

# ╔═╡ 333094ef-f309-4816-9651-df44d51f9d95
md"""
## Multivariate function ``\mathbb{R}^n \rightarrow \mathbb{R}``

"""

# ╔═╡ 056186fb-4db5-46c5-a2df-fdb19684ffcc
begin
	f_demo(w₁, w₂) = 1/4 * (w₁^4 + w₂^4) - 1/3 *(w₁^3 + w₂^3) - w₁^2 - w₂^2 + 4
	f_demo(w::Vector{T}) where T <: Real = f_demo(w...)
	∇f_demo(w₁, w₂) = [w₁^3 - w₁^2 - 2 * w₁, w₂^3 - w₂^2 - 2 * w₂]
	∇f_demo(w::Vector{T}) where T <: Real = ∇f_demo(w...)
end;

# ╔═╡ bbf4a250-96f3-457c-9bb1-f4147bff8056
more_ex_surface = let
	gr()
	plot(-2:0.1:3, -2:0.1:3, f_demo, st=:surface, color=:jet, colorbar=false, aspect_ratio=1.0, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"f(x)", title="A "*L"\mathbb{R}^2\rightarrow \mathbb{R}"*" function", size=(300,300))
end;

# ╔═╡ 8af85d4c-9759-4549-8827-b6e54868aa38
TwoColumn(
md"""


\
\
\
	
Machine learning models are multivariate functions


 
```math
\large 
f(\mathbf{x}): \mathbb R^n \rightarrow \mathbb R
```

* ``n``: the number of inputs


""",
more_ex_surface
	
)

# ╔═╡ a56baf6a-cd46-4eb0-9d98-97b22cbdebee
md"""
## Linear function: ``\mathbb R^n \rightarrow \mathbb R``



```math
\large 

\begin{align}
f(\mathbf{x}) &=   c + b_1 x_1 + b_2 x_2 + \ldots  b_n x_n\\
	&= w_0 + \mathbf{b}^\top \mathbf{x} 
\end{align}
```

* where ``\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top`` 
* ``\mathbf{b} = [w_1, w_2, \ldots, w_n]^\top``


"""

# ╔═╡ 3dfbd08c-9c75-4d9a-8f8c-3b0976b880b3
aside(tip(md"
Recall ``\mathbf{b}^\top \mathbf{x}  = b_1 x_1 + b_2 x_2 + \ldots  b_n x_n``
"))

# ╔═╡ 1619883b-fccc-4e39-a18f-2868bf3d05e5
md"""
##

```math
\large 

\begin{align}
f(\mathbf{x})
	&= w_0 + \mathbf{b}^\top \mathbf{x} 
\end{align}
```
* direct generalisation of the linear function

$\large f(x) = c + b\cdot x$

* from a line to (hyper-)plane
"""

# ╔═╡ 716bbad1-873c-4623-9874-35d5812755b2
md"""

## 


**Example:**
The most boring function a flat plane: when ``\mathbf{b} =\mathbf{0}, c=1``


```math
\large
f(\mathbf{x}) = 1
```

* generalisation of horizontal line ``f(x)=1``

"""

# ╔═╡ 1577f3b9-d4e4-4f90-a2d8-c24a582e3842
let
	gr()
	plot(-10:0.1:10, -10:0.1:10, (x1, x2) -> 1, st=:surface, zlim=[-0.1, 2], xlabel=L"x_1", ylabel=L"x_2", zlabel=L"f", alpha=0.8, c=:jet)
end

# ╔═╡ c418d004-90ad-4fdb-b830-cfca116ef89c
md"""

##

**Example**: a less boring function: (hyper-)plane

```math
\large 
f(\mathbf{x}) = 10 + \begin{bmatrix}1\\ 0\end{bmatrix} \begin{bmatrix}x_1& x_2\end{bmatrix}
```


* *i.e.* ``\mathbf{b} = [1,0]^\top``

* the change of ``x_2`` has no impact on ``f`` (as ``b_2=0``)
"""

# ╔═╡ e878821e-23a1-4de2-b751-f23da4e31023
b = [1, 0]

# ╔═╡ 1e3f91ae-df75-446d-9d10-c08b3174d0e9
let
	plotly()
	b = b
	w₀ = 10
	plot(-15:2:15, -15:2:15, (x1, x2) -> dot(b, [x1, x2])+w₀, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, c=:jet, colorbar=false)
	# plot!(-15:2:15, -10:2:10, (x1, x2) -> 10.0, st=:surface, alpha=0.8, title="Level set")
end

# ╔═╡ efe4a943-c1b5-4ebf-b412-d995b8259335
md"""

## Aside: quadratic form


For **vector** ``\mathbf{x} \in \mathbb{R}^n``, and square matrix ``\mathbf{A} \in \mathbb R^{n\times n}``, the quadratic form


```math
\large
\mathbf{x}^\top \mathbf{A}\mathbf{x}
```

* the end result a **scalar** or **vector** or **matrix** ?
* generalisation of ``x \cdot a \cdot x`` or ``ax^2``


"""

# ╔═╡ 0d635d0c-6b9b-4909-a3bf-96bc51658f40
md"""

## Aside: quadratic form





For **vector** ``\mathbf{x} \in \mathbb{R}^n``, and square matrix ``\mathbf{A} \in \mathbb R^{n\times n}``, the quadratic form


```math
\large
\mathbf{x}^\top \mathbf{A}\mathbf{x}
```

* the end result a **scalar** or **vector** or **matrix** ?
* generalisation of ``x \cdot a \cdot x`` or ``ax^2``


"""

# ╔═╡ d6c13bea-4102-4d21-9cd1-4f04ea3bc110
md"""

## Aside: quadratic form



The quadratic form


```math
\large
\begin{align}
\mathbf{x}^\top \mathbf{A}\mathbf{x} &= \begin{bmatrix} x_1& x_2& \ldots& x_n\end{bmatrix} \begin{bmatrix}
a_{11} & a_{12} & \ldots & a_{1n}\\
a_{21} & a_{22} & \ldots & a_{2n} \\
\vdots & \vdots &\ddots & \vdots \\
a_{n1} & a_{n2} & \ldots & a_{nn}
\end{bmatrix}  \begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}\\
&= \boxed{\sum_{i=1}^n\sum_{j=1}^n a_{ij} x_i  x_j}
\end{align}
```


* the end result is a **scalar**!
"""

# ╔═╡ 7709b758-a81a-4244-8b67-1eb0ec4f178e
Foldable("Why?", md"""

```math

\begin{align}
\mathbf{x}^\top \mathbf{A}\mathbf{x} &=\begin{bmatrix} x_1& x_2& \ldots& x_n\end{bmatrix} \underbrace{\begin{bmatrix}
a_{11} & a_{12} & \ldots & a_{1n}\\
a_{21} & a_{22} & \ldots & a_{2n} \\
\vdots & \vdots &\ddots & \vdots \\
a_{n1} & a_{n2} & \ldots & a_{nn}
\end{bmatrix}  \begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}}_{\begin{bmatrix}\sum_{j} a_{1j} x_j & \sum_{j} a_{2j} x_j & \ldots & \sum_{j} a_{nj} x_j \end{bmatrix}^\top}\\

&=\begin{bmatrix} x_1& x_2& \ldots& x_n\end{bmatrix}\begin{bmatrix}\sum_{j} a_{1j} x_j \\ \sum_{j} a_{2j} x_j \\ \vdots \\ \sum_{j} a_{nj} x_j \end{bmatrix}\\
&= \sum_i x_i \left (\sum_j a_{ij} x_j \right ) \\
&= \sum_i \sum_j  a_{ij} x_i x_j 
\end{align} 
```


""")

# ╔═╡ 18435e02-7911-4971-a9b1-adda47a96b04
md"""

## Example

For ``\mathbf{A} = \mathbf{I}``

```math
\large
\mathbf{x}^\top \mathbf{I}\mathbf{x} = \begin{bmatrix} x_1& x_2\end{bmatrix} \begin{bmatrix}
1 & 0\\
0 & 1 \\
\end{bmatrix}  \begin{bmatrix}x_1 \\ x_2  \end{bmatrix} = x_1^2 + x_2^2
```
* the sum of squares (of all entries): therefore always positive

Generalisation to ``n > 2``, for an error vector ``\mathbf{e} \in \mathbb{R}^n``

```math
\large
\mathbf{e}^\top \mathbf{I}\mathbf{e} = \mathbf{e}^\top\mathbf{e} = \sum_{i=1}^n e_i^2
```

* again, the sum of squares
* squared distance between ``\mathbf{e}`` and ``\mathbf{0}``
"""

# ╔═╡ 8aef8b0f-2c00-4cb2-877a-d11ce8ca3cbc

md"""

## Multi-var. _quadratic_ function: ``\mathbb{R}^n \rightarrow \mathbb{R}``

Recall a single-variate quadratic function

```math
\large
f(x) = x^2
```
"""

# ╔═╡ 560bc22a-d907-4acc-8bd9-6dd45a53a01a
md"""


* it returns the _squared distance_ between ``x`` and ``0``

```math
x^2 = |x-0|^2
```

* ``|x-0|``: error between ``x`` and ``0``

"""

# ╔═╡ 1a93d491-79f9-4bd5-ae9b-37b08f5a0348
@bind x0 Slider(-5:0.5:5; default=0, show_value=true)

# ╔═╡ 8a5f30ad-f686-42d3-b634-32342a16c299
let
	gr()
	# a, b, c = qa_, qb_, qc_
	# x₀ = (a ==0) ? 0 : -b/(2*a)
	# plt = plot((x₀-1.5*abs(2a)):0.1:(x₀+1.5*abs(2a)), (x) -> a*x^2+b*x + c, framestyle=:origin, label=L"f(x) = %$(a)x^2 + %$(b)x + %$(c)", legend=:outerright, lw=2)

	# # abcs = [(-2, 0, -2), (-2, 3, 1)]
	# for (a, b, c) in abcs
	# 	plot!(-4:0.1:4, (x) -> a*x^2+b*x+c, framestyle=:origin, lw =2, label=L"f(x) = %$(a)x^2 + %$(b)x + %$(c)", legend=:outerright)
	# end


	plot((x)->x^2, label=L"f(x) = x^2", lw=2, framestyle=:origin, title=L"f(x)=x^2")
	x_ = x0
	plot!([x_, x_], [0, x_^2], ls=:dash, lc=:gray, lw=2, label="")

	annotate!([x_], [x_^2/2], L"x^2= %$(x_^2)", :right)
	# plt
end

# ╔═╡ a26408c3-1d2c-48af-b373-c87098c364de

md"""

## Multi-var. _quadratic_ function: ``\mathbb{R}^n \rightarrow \mathbb{R}``

Its two variate counter part for ``\mathbf{x} =[x_1, x_2]^\top``

```math
\large
f(x) = x^2 \textcolor{blue}{\xRightarrow{\rm generalisation}} f(\mathbf{x}) = x_1^2 + x_2^2 
```
"""

# ╔═╡ 1bdd7aec-3bcf-4130-8a81-2acf45437e3f
md"""

## Multi-var. _quadratic_ function: ``\mathbb{R}^n \rightarrow \mathbb{R}``

Its multi-variate counter part for ``\mathbf{x}\in \mathbb{R}^n``

```math
\large
f(x) = x^2 \textcolor{blue}{\xRightarrow{\rm generalisation}} f(\mathbf{x})= \mathbf{x}^\top\mathbf{x}
```
"""

# ╔═╡ ba90e665-7f2b-44a5-9a8c-587249481272
md"""

* it still returns the _squared distance_ between ``\mathbf{x}`` and ``\mathbf{0}``

```math
\mathbf{x}^\top \mathbf{x} = (\mathbf{x} -\mathbf{0})^\top (\mathbf{x} -\mathbf{0})
```


"""

# ╔═╡ f253f70e-4fe4-489d-ab40-4fd222eaa413
md"move me: $(@bind x0_ Slider(-6:0.1:6, default=0.8))"

# ╔═╡ b96382d6-1837-4db7-aeb6-0d0bc868205b
v0 = [1, 1]

# ╔═╡ c4b26533-1fd8-4a5a-88ed-f767cc0b765b
let
	plotly()
	x0 = [0, 0]
	A = Matrix(I,2,2)
	μ = [0,0]
	f(x1, x2) = dot([x1, x2]- μ, A, [x1, x2]-μ)
	plot(μ[1]-5:1:μ[1]+5, μ[2]-5:1:μ[2]+5, f, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.5, color=:jet, framestyle=:zerolines, ratio=1, colorbar=false, xlim=[-6, 6], ylim = [-6, 6], title="Qudratic function xᵀx")
	scatter!([x0[1]], [x0[2]], [0], ms=1, markershape=:cross, label="")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=1, label="")
	vs = 0:0.1:f(x0...)
	plot!(2 * ones(length(vs)), 2 * ones(length(vs)), vs, lw=3, ls=:dash, lc=:gray, label="")
	
	ys = -5:.5:5
	xs = x0[1] * ones(length(ys))
	zs = [dot([xs[i], ys[i]]- μ, A, [xs[i], ys[i]]-μ) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]]- μ, A, [ys[i], xs[i]]-μ) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=3, label="", c=1)
	path3d!(ys, xs,zs2, lw=3, label="", c=2)

	x_ = x0_ * v0 / (norm(v0))
	plot!([x_[1], x_[1]], [x_[2], x_[2]], [f(x_...), 0], lw=3, lc=:black, ls=:dashdot,label="")

	scatter!([x_[1]], [x_[2]], [0], ms=2, markershape=:cross, label="x")
	scatter!([x_[1]], [x_[2]], [f(x_...)], ms=1, markershape=:circle, label="")
end

# ╔═╡ 156378ef-d2f6-452a-8d76-59f697f03c17
md"""

## Multi-var quadratic function: ``\mathbb{R}^n \rightarrow \mathbb{R}``


More generally, the multi-variate quadratic function for ``\mathbf{x}\in \mathbb{R}^n``

```math
\Large
f(x) = a x^2 + bx + c \textcolor{blue}{\xRightarrow{\rm generalisation}} f(\mathbf{x})= \mathbf{x}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{x} + c
```



* ``\mathbf{x}^\top \mathbf{A}\mathbf{x}``: is called quadratic form
"""

# ╔═╡ d99bce21-59ca-491e-a7a2-57ad4abe73ed
md"""

## Recall the max/min test


The quadratic coefficient ``a`` determines: maximum, minimum
"""

# ╔═╡ 814a51b4-bfb7-4dcf-953c-c5f9e20f853f
pltapos, pltaneg=let
	gr()
	b, c = 0, 0
	plt = plot( legend=:outerbottom, title="Effect of "*L"a>0", size=(300,400))
	plt2 = plot( legend=:outerbottom, title="Effect of "*L"a<0", size=(300,400))
	
	ass = [0.1, 1,2,3,4,6]
	for a in ass
		plot!(plt, -5:0.2:5, (x) -> a* x^2 + b* x+ c, framestyle=:origin, label=L"f(x) = %$(a)x^2 + %$(b)x + %$(c)", lw=2)
		plot!(plt2, -5:0.2:5, (x) -> -a * x^2 + b* x+ c, framestyle=:origin, label=L"f(x) = -%$(a)x^2 + %$(b)x + %$(c)", lw=2)
	end


	plt, plt2
end;

# ╔═╡ af66f43d-27be-4a8c-bd4b-60e25e113214
TwoColumn(md"

#### when `` a > 0``


The function has a **minimum**

$(pltapos)
", 
	
	
md" #### when `` a<0``


The function has a **maximum**


$(pltaneg)
")

# ╔═╡ 92a4d83a-7637-4201-9c5e-c36b379210a0
md"""

## Max/min test: multivariate function
```math
\Large
f(\mathbf{x})= \mathbf{x}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{x} + c
```

* cross reference single variate case (with a minimum)

$a>0, ax^2 > 0\; \text{for all } {x\in \mathbb{R}}$
* when ``\mathbf{A}`` is _positive definite_, *i.e.* when 
$\text{positive definite: }\large \mathbf{x}^\top\mathbf{A}\mathbf{x} > 0\; \text{for all } \mathbf{x\in \mathbb{R}^n}$

> **Interpretation**:  _for all directions ``\mathbf{x}\in \mathbb{R}^n``, ``f`` **faces UP**_
* then ``f`` has a **minimum**

* *e.g.*
```math
\mathbf{A} = \begin{bmatrix}1 & 0 \\0 & 1\end{bmatrix}
```



"""

# ╔═╡ 602e6920-54bd-4579-9b5d-63e477d1431e
md"""

## Max/min test
```math
\Large
f(\mathbf{x})= \mathbf{x}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{x} + c
```

* cross reference for singular variate case 

$a<0, ax^2 < 0\; \text{for all } {x\in \mathbb{R}}$
* when ``\mathbf{A}`` is _**negative definite**_, *i.e.* when 

$\text{negative definite: }\large \mathbf{x}^\top\mathbf{A}\mathbf{x} < 0\; \text{for all } \mathbf{x\in \mathbb{R}^n}$

> **Interpretation**:  _for all directions ``\mathbf{x}\in \mathbb{R}^n``, ``f`` **faces DOWN**_

* then ``f`` has a **maximum**
* *e.g.*
```math
\mathbf{A} = \begin{bmatrix}-1 & 0 \\0 & -1\end{bmatrix}
```


"""

# ╔═╡ 3e6030a0-81e6-43a7-8bc6-beb6ef9aab87
md"""

## Max/min test
```math
\Large
f(\mathbf{x})= \mathbf{x}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{x} + c
```

* when ``\mathbf{A}`` is _**NOT definite**_, *i.e.* when 

$\large \mathbf{x}^\top\mathbf{A}\mathbf{x} > 0\;\; \text{or} \;\;\mathbf{x}^\top\mathbf{A}\mathbf{x} < 0$

* *e.g.*
```math
\mathbf{A} = \begin{bmatrix}1 & 0 \\0 & -1\end{bmatrix}
```

* ``f`` is neither maximum or minimum: not determined

* this is known as a saddle point
  * it has a maximum in one direction but minimum in the other 
"""

# ╔═╡ ac006f54-7108-459c-a800-6832b2de254d
md"""

## Level set and contour plot


> **Level set**:  a curve in the input space of the same height ``h``: 
> 
> $\large \{\mathbf{x} \in \mathbb R^m| f(\mathbf{x}) = h\}$



* intuitively, we intersect the surface with a _horizontal_ plane of height ``h``
* the **intersected curve** is a level set


**Contour plot**: the plot of different level sets in the input plane
"""

# ╔═╡ 22ca5cae-7ea7-47a4-b551-dcef2158f4ea
html"""<center><img src="https://i.stack.imgur.com/tHSPs.gif" width = "350"/></center>""" 

# ╔═╡ 80a793f7-9c8c-493c-afcc-4beba2f0bf21
md"""

## Contour plot of linear function
"""

# ╔═╡ 25f5ce83-c520-415d-9537-3a483a20e178
md""" Height ``h``: $(@bind height Slider(-5:1.0:23, default =5, show_value=true))"""

# ╔═╡ b2e5c268-8dfa-425e-9abf-b3ec50e9c0c5
bv = [1, 0];

# ╔═╡ ae351013-0928-4544-b043-54091f17565f
begin 
	c₀ = 10
end;

# ╔═╡ 355d7e19-8e92-4ce0-afbd-854ea3a3f33e
let
	plotly()
	b = bv
	b₀ = c₀
	plt = plot(-15:1:15, -10:1:10, (x1, x2) -> dot(b, [x1, x2])+b₀, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, c=:jet, colorbar=false)

	plot!(-15:2:15, -10:2:10, (x1, x2) -> height, st=:surface, alpha=0.8, c=:gray, title="Level set")
	plt
end

# ╔═╡ 64834d5f-cb72-4ab4-81ba-7a3d7eb8d8c0
linear_f(x; b = bv, c= c₀) = dot(b, x) + c;

# ╔═╡ 7cebb1ad-bddf-41b0-8d50-c8a2dc2bd202
let
	gr()
	f(x) = linear_f(x)
	# b = b
	# ∇f(x₁, x₂) = b * 3
	xs = -15:0.5:15
	ys= -15:0.5:15
	cont = contour(xs, ys, (x, y)->f([x,y]), c=:jet, xlabel=L"x_1", ylabel=L"x_2", title="Contour plot of hyperplane")
	# for better visualisation
	# meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.
	# xs_, ys_ = meshgrid(range(-15, 15, length=4), range(-15, 15, length=4))
	# quiver!(xs_, ys_, quiver = ∇f, c=:green)
end

# ╔═╡ 087d8ccb-0207-4dad-ac44-d25a4835de4f
md"""
## Contour plot of quadratic function


**Example**: quadratic function

```math
\large 
f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2 
```


* generalisation of ``f(x) = x \cdot x = x^2``
"""

# ╔═╡ 4b899664-f5b9-4932-a3bb-38f818efb60d
md""" Height: $(@bind height_ Slider(0:1.0:30, default =5, show_value=true))"""

# ╔═╡ 01c290ac-d448-47bd-ae74-545c6c4d2c00
let
	plotly()
	x0 = [0, 0]
	A = Matrix(I,2,2)
	μ = [0,0]
	f(x1, x2) = dot([x1, x2]- μ, A, [x1, x2]-μ)
	plot(μ[1]-5:1:μ[1]+5, μ[2]-5:1:μ[2]+5, f, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.5, color=:jet, framestyle=:zerolines, ratio=1, colorbar=false, xlim=[-6, 6], ylim = [-6, 6])
	scatter!([x0[1]], [x0[2]], [0], ms=1, markershape=:cross, label="")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=1, label="")
	vs = 0:0.1:f(x0...)
	plot!(2 * ones(length(vs)), 2 * ones(length(vs)), vs, lw=3, ls=:dash, lc=:gray, label="")
	
	plot!(-15:2:15, -10:2:10, (x1, x2) -> height_, st=:surface, alpha=0.8, c=:gray, title="Level set and contour")

	t = range(0, stop=2π, length=1000)
	height = height_
	x = cos.(t) * sqrt(height)
	y = sin.(t) * sqrt(height)
	z = ones(length(t)) * height
	plot!(x, y, z, lw=4, lc=2, label="")
	
	# ys = -5:.5:5
	# xs = x0[1] * ones(length(ys))
	# zs = [dot([xs[i], ys[i]]- μ, A, [xs[i], ys[i]]-μ) for i in 1:length(ys)]
	# zs2 = [dot([ys[i], xs[i]]- μ, A, [ys[i], xs[i]]-μ) for i in 1:length(ys)]
	# path3d!(xs,ys,zs, lw=3, label="", c=1)
	# path3d!(ys,xs,zs2, lw=3, label="", c=2)

	# x_ = x0_ * v0 / (norm(v0))
	# plot!([x_[1], x_[1]], [x_[2], x_[2]], [f(x_...), 0], lw=3, lc=:black, ls=:dashdot,label="")

	# scatter!([x_[1]], [x_[2]], [0], ms=2, markershape=:cross, label="x")
	# scatter!([x_[1]], [x_[2]], [f(x_...)], ms=1, markershape=:circle, label="")
end

# ╔═╡ 788e7c1b-d58d-4553-bce6-eaa4cc3085ae
md"""

##
"""

# ╔═╡ 1a8ac980-8a88-47f3-ac23-3333945c5720
let
	gr()
	A = Matrix(I, 2, 2)
	f(x₁, x₂) = dot([x₁, x₂], A, [x₁, x₂])
	# ∇f(x₁, x₂) = 2 * A* [x₁, x₂] / 5
	xs = -20:0.5:20
	ys= -20:0.5:20
	contour(xs, ys, (x, y)->f(x,y), c=:jet, xlabel=L"x_1", ylabel=L"x_2", title="Contour plot of " * L"f(\mathbf{x})=\mathbf{x}^\top\mathbf{x}", ratio=1, xlim=[-20,20])
	# for better visualisation
	# meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.
	# xs_, ys_ = meshgrid(range(-15, 15, length=8), range(-15, 15, length=6))
	# quiver!(xs_, ys_, quiver = ∇f, c=:green)
end

# ╔═╡ 313cc81a-3568-40f5-8b3f-76d0ee36abfa
md"""
## Partial derivative

Multivariate function $f(\mathbf{x}): R^n\rightarrow R$, **partial derivative** w.r.t. $x_i$ is

$$
\large \frac{\partial f}{\partial x_i}(\mathbf{x}) = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, \textcolor{red}{x_i+h}, \ldots, x_n) - f(x_1, \ldots, \textcolor{red}{x_i}, \ldots, x_n)}{h}$$

* _change_ one dimension (``i-`` th dimension) each time **while keeing** all ``x_j`` constant, ``j\neq x``


"""

# ╔═╡ c2efcc3d-b5be-4047-ad7d-e80e9186f7f9
md"""

## Partial derivative

Multivariate function $f(\mathbf{x}): R^n\rightarrow R$, **partial derivative** w.r.t. $x_i$ is

$$
\large \frac{\partial f}{\partial x_i}(\mathbf{x}) = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, \textcolor{red}{x_i+h}, \ldots, x_n) - f(x_1, \ldots, \textcolor{red}{x_i}, \ldots, x_n)}{h}$$

* _change_ one dimension (``i-`` th dimension) each time **while keeing** all ``x_j`` constant, ``j\neq x``

This is the same as
```math
\large
\frac{\partial f}{\partial x_i}(\mathbf{x}) = \lim_{h \rightarrow 0} \frac{f(\mathbf{x}+ h \cdot \textcolor{red}{\mathbf{e}_i}) - f(\mathbf{x})}{h}

```

* ``\mathbf{e}_i`` is the i-th standard basis vector, as an example, for ``i=1``
```math
\mathbf{x} + h\cdot  \mathbf{e}_1 =\begin{bmatrix}x_1  \\ x_2 \\\vdots \\ x_n \end{bmatrix} + h \begin{bmatrix}1  \\ 0 \\\vdots \\ 0 \end{bmatrix}= \begin{bmatrix}x_1 + h \\ x_2 \\\vdots \\ x_n \end{bmatrix}
```



- change rate (slope) along the ``i``-th standard basis (``\textcolor{red}{\mathbf{e}_i}``) direction 


"""

# ╔═╡ 3d1c6668-9d2b-46a4-82bf-0dae728dfe8b
md"""

## Example


As a concrete example, for two-variate input, and ``i=1``
```math
\large
\begin{align}
\frac{\partial f}{\partial x_1}(\mathbf{x})& = \lim_{h \rightarrow 0} \frac{f(x_1+h, x_2) - f(x_1, x_2)}{h}\\
&= \lim_{h \rightarrow 0} \frac{f(\mathbf{x} + h\cdot \textcolor{red}{\mathbf{e}_1}) - f(\mathbf{x})}{h}
\end{align}
```

* *say* ``\mathbf{x}= [2,2]^\top`` and ``x_2=2`` is fixed (the ``\textcolor{red}{\rm red\; curve}``)
* the partial is red curve's derivative along the **curve**

"""

# ╔═╡ 547e2772-5a5f-4cc9-bec2-a9c67fed8b5a
@bind add_theother CheckBox(default=false)

# ╔═╡ 5b363889-b66e-4e10-8e51-78791510de48
xx = [2,2]

# ╔═╡ 8a781eb7-e0db-4286-98e8-70a3beb49952
md"""
## Gradient


**Gradient** is the collection of the partial derivatives:

$$\large \nabla f(\mathbf{x})=\text{grad} f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\\ \vdots \\ \frac{\partial f(\mathbf{x})}{\partial x_n}\end{bmatrix}$$


- **gradient** itself is a (vector to vector) function! 
  - **input**:  a vector ``\mathbf{x} \in R^n`` (interpreted as an input **location**) 
  - **output**: a vector ``\nabla f(\mathbf{x})`` (interpreted as a **direction**)


"""

# ╔═╡ 6422fa63-d113-45f5-aee7-8df4836138d5
html"""<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/3d-gradient-cos.svg/2560px-3d-gradient-cos.svg.png" width = "350"/></center>""" 

# ╔═╡ 9279bed6-cade-4ae7-9441-81facec7ee1c
md"""
## Gradient direction


A key fact of **gradient**:

!!! infor "Gradient direction"
	Gradient vector ``\nabla f(\mathbf{x}_0)`` points to 
	* the **greatest ascent direction** locally at location ``\mathbf{x}_0``.
"""

# ╔═╡ 17a01f79-4791-4c09-82d9-369e5bd58b10
html"""<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/3d-gradient-cos.svg/2560px-3d-gradient-cos.svg.png" width = "350"/></center>""" 

# ╔═╡ 7bdccd4e-b9e4-4571-b086-b98619022bf1
md"[source](https://en.wikipedia.org/wiki/Gradient)"

# ╔═╡ 159948ef-588d-4ecd-80e4-dcd335d5d4e7
md"""

## Examples: ``f(\mathbf{x}) = \mathbf{b}^\top \mathbf{x} + c``


For linear function, ``\mathbf{x} =[x_1, x_2]^\top \in \mathbb R^2``

```math
\large
f(\mathbf{x}) =\mathbf{b}^\top \mathbf{x} + c = b_1 x_1 + b_2 x_2 + c
```

The gradient is 

$$\large

\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} b_1\\ b_2\end{bmatrix} = \mathbf{b}$$



The gradient function outputs _**constant**_ direction ``\mathbf{b}`` for all locations ``\mathbf{x} \in \mathbb R^2``
* similar to uni-variate linear function: the slope is also a constant
*  ``\mathbf{b}``: the greatest ascend direction
"""

# ╔═╡ 6490d76d-0b7b-4720-91b2-7e05fbfae0f9
bvec = [1, 0]

# ╔═╡ 3432bede-1139-4ae8-a292-9fe3bffdf9b3
let
	plotly()
	b = bvec
	b₀ = c₀
	plot(-15:1:15, -10:1:10, (x1, x2) -> dot(b, [x1, x2])+b₀, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, c=:jet, colorbar=false)
end

# ╔═╡ 2b2f840a-95a5-4882-a2b6-79e12e02d805
let
	gr()
	f(x) = linear_f(x; b=bvec)
	∇f(x₁, x₂) = bvec * 3
	xs = -15:0.5:15
	ys= -15:0.5:15
	cont = contour(xs, ys, (x, y) -> f([x,y]), c=:jet, xlabel=L"x_1", ylabel=L"x_2", title="Contour and gradients of "*L"f(\mathbf{x}) = %$(bvec)^\top \mathbf{x}", framestyle=:origin)
	# for better visualisation
	meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.
	xs_, ys_ = meshgrid(range(-15, 15, length=4), range(-15, 15, length=4))
	quiver!(xs_, ys_, quiver = ∇f, c=:green)
end

# ╔═╡ 2d72ed24-21f9-4a4b-9fcf-e20d83a5a1c1
let
	gr()
	bvec = [1,1]
	f(x) = linear_f(x; b=bvec)
	∇f(x₁, x₂) = bvec * 3
	xs = -15:0.5:15
	ys= -15:0.5:15
	cont = contour(xs, ys, (x, y) -> f([x,y]), c=:jet, xlabel=L"x_1", ylabel=L"x_2", title="Contour and gradients of "*L"f(\mathbf{x}) = %$(bvec)^\top \mathbf{x}", framestyle=:origin)
	# for better visualisation
	meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.
	xs_, ys_ = meshgrid(range(-15, 15, length=4), range(-15, 15, length=4))
	quiver!(xs_, ys_, quiver = ∇f, c=:green)
end

# ╔═╡ 9da5feca-40f5-49c7-939f-49c6266970e6
md"""

## Examples: ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}``


For quadratic function, ``\mathbf{x} =[x_1, x_2]^\top \in \mathbb R^2``

```math
\large 
f(\mathbf{x}) =\mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2
```

The gradient is 

$$
\large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} 2 x_1\\ 2x_2\end{bmatrix} = 2 \mathbf{x}$$



The gradient function outputs: ``2\mathbf{x}``

* the gradient vanishes when ``\mathbf{x} = \mathbf{0}``
* pointing outwardly (the greatest ascend direction)
"""

# ╔═╡ 58c7aabe-b3ae-4bc8-ba61-356f7e0fa31d
TwoColumn(let
	x0 = [0, 0]
	plotly()
	A = Matrix(I,2,2)
	μ = [0,0]
	f(x1, x2) = dot([x1, x2]- μ, A, [x1, x2]-μ)
	plot(μ[1]-5:1:μ[1]+5, μ[2]-5:1:μ[2]+5, f, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.5, color=:jet, framestyle=:zerolines, ratio=1, colorbar=false, xlim=[-6, 6], ylim = [-6, 6], size=(300,300))
	scatter!([x0[1]], [x0[2]], [0], ms=1, markershape=:cross, label="")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=1, label="")
	vs = 0:0.1:f(x0...)
	plot!(2 * ones(length(vs)), 2 * ones(length(vs)), vs, lw=3, ls=:dash, lc=:gray, label="")

end, let
	gr()
	A = Matrix(I, 2, 2)
	f(x₁, x₂) = dot([x₁, x₂], A, [x₁, x₂])
	∇f(x₁, x₂) = 2 * A* [x₁, x₂] / 5
	xs = -20:0.5:20
	ys= -20:0.5:20
	cont = contour(xs, ys, (x, y)->f(x,y), c=:jet, xlabel=L"x_1", ylabel=L"x_2", framestyle=:origin, title="Gradient field plot", ratio=1, size=(300,300))
	# for better visualisation
	meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.
	xs_, ys_ = meshgrid(range(-15, 15, length=8), range(-15, 15, length=6))
	quiver!(xs_, ys_, quiver = ∇f, c=:green)
end)

# ╔═╡ 9e9c6830-7544-4d81-95c6-34dca751f75f
md"""

!!! note "Observation"
	Note the gradients "vanish" (smaller arrows mean smaller scale) near the centre (*i.e.* stationary points)
    * no surprise $$\nabla f(\mathbf{x}) =  \begin{bmatrix} 2 x_1\\ 2x_2\end{bmatrix}$$, the output vector gets smaller when ``\mathbf{x}\rightarrow \mathbf{0}``


"""

# ╔═╡ 4a5b5744-701e-4051-ac88-b2c3d9715413
md"""

## How about ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{Ax} + \mathbf{b}^\top \mathbf{x} + c``


!!! question "Question"

	What's the gradient of the full (multivariate) quadratic function?

	```math
	f(\mathbf{x}) = \mathbf{x}^\top \mathbf{Ax} + \mathbf{b}^\top \mathbf{x} + c
	```


!!! note "Hint"
	Recall univariate quadratic function and its derivative:
	
	```math
	f(x) = ax^2+bx+c,\;\; f'(x) = 2ax + b
	```

#### Just guess?

"""

# ╔═╡ 1c5210c8-7f50-4a6f-a37b-5a6b19b25fc7
Foldable("Answer", md"

```math
\large
\nabla f(\mathbf{x}) = (\mathbf{A} +\mathbf{A}^\top)\mathbf{x} + \mathbf{b}
```


For symmetric ``\mathbf{A}``, the result is

```math
\large
\boxed{\nabla f(\mathbf{x}) = 2 \mathbf{A}\mathbf{x} + \mathbf{b}}
```

Check appendix for full justification. 
")

# ╔═╡ 749ee0b8-7cf5-4c44-9029-08732d959958
md"""

## Summary of some gradients



Multivariate generalisations:

* constant function: ``f(\mathbf{x}) =c``
* linear hyperplane function: ``f(\mathbf x) = \mathbf{b}^\top \mathbf x+ c``
* quadratic function: ``f(\mathbf x) = \mathbf x^\top \mathbf{A}\mathbf{x} + \mathbf{b}^\top \mathbf x+ c``


**Gradient** of the above three: 


```math
\large
\begin{align}
f(\mathbf{x}) =c &,\;\; \nabla f(\mathbf{x}) = \mathbf{0}\\
f(\mathbf{x}) =\mathbf{b}^\top \mathbf x+ c &,\;\; \nabla f(\mathbf{x}) =\mathbf{b}\\
f(\mathbf{x}) =\mathbf x^\top \mathbf{A}\mathbf{x} + \mathbf{b}^\top \mathbf x+ c &,\;\; \nabla f(\mathbf{x}) = 2\mathbf{Ax} + \mathbf{b}
\end{align}
```

* assume ``\mathbf{A}`` is symmetric
"""

# ╔═╡ 2f816b4e-ce78-4bc9-9a57-75961e944c78
md"""
## Optimise multivariate functions



Often, we want to optimise a ``\mathbb R^n \rightarrow \mathbb R`` function ``f(\mathbf{x})``


```math
\large
\hat{\mathbf{x}} \leftarrow \arg\min_{\mathbf{x}} f(\mathbf{x})
```

* the idea is the same, we are looking for ``\hat{\mathbf{x}}`` where the **gradient** is zero, *i.e.* 


```math
\large
\nabla f(\mathbf{x}) = \mathbf{0}
```


"""

# ╔═╡ cc6fa932-1e64-4379-b4b1-5d4096fbcb7e
md"""

## Optimise quadratic form -- closed form solution


Recall that for quadratic function ``f(x) = ax^2+bx+c`` and ``f'(x) = 2ax+b``


* the stationary point is

```math
\large
f'(x) = 2ax+b =0 \Rightarrow \boxed{x' = -\frac{1}{2}a^{-1}b}
```

	"""

# ╔═╡ 0488519b-a1cd-40db-8ee9-d2658b576bd2

md"""

## Optimise quadratic form -- closed form solution


Recall that for quadratic function ``f(x) = ax^2+bx+c`` and ``f'(x) = 2ax+b``


* the stationary point is

```math
\large
f'(x) = 2ax+b =0 \Rightarrow \boxed{x' = -\frac{1}{2}a^{-1}b}
```

Similarly, for multivariate quadratic function with symmetric ``\mathbf{A}``:

```math
\large
f(\mathbf{x}) = \mathbf{x}^\top \mathbf{Ax} + \mathbf{b}^\top \mathbf{x} + c
```

Its gradient is

```math
\large
\nabla f(\mathbf{x}) = 2 \mathbf{A}\mathbf{x} + \mathbf{b}
```


The stationary point is


```math
\large
\nabla f(\mathbf{x}) = 2 \mathbf{A}\mathbf{x} + \mathbf{b} =\mathbf{0} \Rightarrow \boxed{\mathbf{x}'= -\frac{1}{2}\mathbf{A}^{-1}\mathbf{b} }
```


"""

# ╔═╡ 3f56e844-6b96-45a7-ad01-bc76d7951979
md"""

## Demonstration

"""

# ╔═╡ 62299cf7-0109-4f26-b140-22421a6e249d
begin
	# A_ = Matrix(I, 2, 2)
	A_ = - Matrix(I, 2, 2)
	# A_ = Matrix([1 0; 0 -1])
end

# ╔═╡ 0ef53efe-603d-4ca5-936f-996be0caf72a
bv_, cv_= 10 * rand(2), 0.0

# ╔═╡ 42507d15-f0b4-480c-95a6-5cc3187684d7
qform(x; A=A_, b=bv_, c=c_) = x' * A * x + b'* x + c 

# ╔═╡ 50982a4f-acae-407a-b7b4-6a1ab653eeec
let
	gr()
	plot(-5:0.5:5, -5:0.5:5, (x,y) -> qform([x,y]; A= -Matrix(I,2,2), b=zeros(2), c=0), st=:surface, colorbar=false, color=:jet, title="A is negative definite; maximum")
end

# ╔═╡ 0696c9bd-2ecf-4cf8-8881-b76730814342
TwoColumn(let 
	plotly()
	A = [1 0; 0 -1]
	plot(-5:0.5:5, -5:0.5:5, (x,y) -> qform([x,y]; A= A, b=zeros(2), c=0), st=:surface, colorbar=false, color=:jet, title="A is not definite; not determined", size=(400,400))
	ys = -5:.5:5
	xs = zeros(length(ys))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=8, label="", c=1)
	xs = -5:.5:5
	ys = zeros(length(xs))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=8, label="", c=2)
end, html"""<center><img src="https://saddlemania.com/wp-content/uploads/2022/04/parts-of-saddle.jpeg" height = "300"/></center>""" )

# ╔═╡ 907d88dd-e475-4922-a6fc-1c296721dc78
let
	plotly()
	xmin_0 = -0.5 * A_^(-1) * bv_
	xs = range(xmin_0[1]-8, xmin_0[1]+8, 100)
	plot(xs, -8:0.8:8, (x1, x2) -> qform([x1, x2]; A=A_, b= bv_, c= cv_), st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, ratio=1, c=:jet, colorbar=false)

	scatter!([xmin_0[1]],[xmin_0[2]], [qform(xmin_0; A=A_, b= bv_, c= cv_)], label="x': min/max/station")
end

# ╔═╡ 0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
md"""

## Appendix
"""

# ╔═╡ 8687dbd1-4857-40e4-b9cb-af469b8563e2
function perp_square(origin, vx, vy; δ=0.1) 
	x = δ * vx/sqrt(norm(vx))
	y = δ * vy/sqrt(norm(vy))
	xyunit = origin+ x + y
	xunit = origin + x
	yunit = origin +y
	Shape([origin[1], xunit[1], xyunit[1], yunit[1]], [origin[2], xunit[2], xyunit[2], yunit[2]])
end

# ╔═╡ fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
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

# ╔═╡ 22cd0ef4-29ad-4118-92f4-c4d440612db5
let
	plotly()
	f(x) = qform(x; A=Matrix(I,2,2), b=zeros(2), c=0) +5
	plt = plot(-5:0.5:5, -5:0.5:5, (x,y) -> f([x,y]), st=:surface, colorbar=false, color=:jet,alpha=0.8, xlim=[-5, 5] , ylim=[-5, 5],zlim =[0, 55], title=L"\mathbf{A}" * "is positive definite; minimum")


	θs = range(0, 2π, 15)
	length = 4
	for (ind, θ) in enumerate(θs)
		x, y = cos(θ) * length, sin(θ)* length	
		arrow3d!([0], [0], [0], [x], [y], [0]; as=0.1, lc=ind, la=0.9, lw=2, scale=:identity)
		v = [cos(θ), sin(θ)]
		xs = range(-5, 5, 50)
		k = v[2]/v[1]
		ys = k .* xs
		zs = [f([x, ys[i]]) for (i, x) in enumerate(xs)]
		path3d!(xs, ys, zs, lw=3, label="", c=ind)
	end
	plt
end

# ╔═╡ f268a7cd-c193-45ae-8cee-a4a0710a2a3a
let
	x0 = xx
	plotly()
	A = Matrix(I,2,2)
	μ = [0,0]
	f(x1, x2) = dot([x1, x2]- μ, A, [x1, x2]-μ) + .5
	plt = plot(μ[1]-5:0.8:μ[1]+5, μ[2]-5:0.8:μ[2]+5, f, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.5, color=:jet, framestyle=:zerolines, ratio=1, colorbar=false, xlim=[-6,6], ylim=[-6, 6])
	scatter!([x0[1]], [x0[2]], [0], ms=1, markershape=:x, label="x'")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=1, label="")
	vs = 0:0.1:f(x0...)
	plot!(x0[1] * ones(length(vs)), x0[2] * ones(length(vs)), vs, lw=3, ls=:dash, lc=:gray, label="")
	arrow3d!([x0[1]], [x0[2]], [0], [3], [0], [0]; as = 0.4, lc=2, la=1, lw=2, scale=:identity)

	xs = -5:0.5:5
	ys = x0[2] * ones(length(xs))
	zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
	path3d!(xs, ys, zs, lw=3, label="", c=2)

	if add_theother
		arrow3d!([x0[1]], [x0[2]], [0], [0], [3], [0]; as = 0.4, lc=1, la=1, lw=2, scale=:identity)

		ys = -5:0.5:5
		xs = x0[1] * ones(length(xs))
		zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
		path3d!(xs,ys,zs, lw=3, label="", c=1)
	end
	plt
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.21"
Plots = "~1.38.15"
PlutoTeachingTools = "~0.2.11"
PlutoUI = "~0.7.51"
Zygote = "~0.6.62"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "01d81ea555e48c0dcc037654a0aa31e03c47e32d"

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

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "0266ee4ffeeac8405ab07c49252c144616fe825d"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.49.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

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

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

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

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

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
git-tree-sha1 = "158232a81d43d108837639d3fd4c66cc3988c255"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.14.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

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
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "0dbc906e66a5e337598dda85f1bfaa81a88251fd"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.7.0"

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

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

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

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

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

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa4b74f80b01c6bc2b89992b861b5f210e665b5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.21+0"

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

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a5aef8d4a6e8d81f171b2bd4be5265b01384c74c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.10"

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
git-tree-sha1 = "ceb1ec8d4fbeb02f8817004837d924583707951b"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.15"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

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

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

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

    [deps.Unitful.extensions]
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

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
git-tree-sha1 = "5be3ddb88fc992a7d8ea96c3f10a49a7e98ebc7b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.62"

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
# ╟─9f90a18b-114f-4039-9aaf-f52c77205a49
# ╟─959d3f6e-ad5b-444f-9000-825063598837
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─653ab1b9-0839-4217-8301-f326efa2a2ad
# ╟─333094ef-f309-4816-9651-df44d51f9d95
# ╟─8af85d4c-9759-4549-8827-b6e54868aa38
# ╟─bbf4a250-96f3-457c-9bb1-f4147bff8056
# ╟─056186fb-4db5-46c5-a2df-fdb19684ffcc
# ╟─a56baf6a-cd46-4eb0-9d98-97b22cbdebee
# ╟─3dfbd08c-9c75-4d9a-8f8c-3b0976b880b3
# ╟─1619883b-fccc-4e39-a18f-2868bf3d05e5
# ╟─716bbad1-873c-4623-9874-35d5812755b2
# ╟─1577f3b9-d4e4-4f90-a2d8-c24a582e3842
# ╟─c418d004-90ad-4fdb-b830-cfca116ef89c
# ╠═e878821e-23a1-4de2-b751-f23da4e31023
# ╟─1e3f91ae-df75-446d-9d10-c08b3174d0e9
# ╟─efe4a943-c1b5-4ebf-b412-d995b8259335
# ╟─0d635d0c-6b9b-4909-a3bf-96bc51658f40
# ╟─d6c13bea-4102-4d21-9cd1-4f04ea3bc110
# ╟─7709b758-a81a-4244-8b67-1eb0ec4f178e
# ╟─18435e02-7911-4971-a9b1-adda47a96b04
# ╟─8aef8b0f-2c00-4cb2-877a-d11ce8ca3cbc
# ╟─560bc22a-d907-4acc-8bd9-6dd45a53a01a
# ╟─1a93d491-79f9-4bd5-ae9b-37b08f5a0348
# ╟─8a5f30ad-f686-42d3-b634-32342a16c299
# ╟─a26408c3-1d2c-48af-b373-c87098c364de
# ╟─1bdd7aec-3bcf-4130-8a81-2acf45437e3f
# ╟─ba90e665-7f2b-44a5-9a8c-587249481272
# ╟─f253f70e-4fe4-489d-ab40-4fd222eaa413
# ╟─b96382d6-1837-4db7-aeb6-0d0bc868205b
# ╟─c4b26533-1fd8-4a5a-88ed-f767cc0b765b
# ╟─156378ef-d2f6-452a-8d76-59f697f03c17
# ╟─d99bce21-59ca-491e-a7a2-57ad4abe73ed
# ╟─af66f43d-27be-4a8c-bd4b-60e25e113214
# ╟─814a51b4-bfb7-4dcf-953c-c5f9e20f853f
# ╟─92a4d83a-7637-4201-9c5e-c36b379210a0
# ╟─22cd0ef4-29ad-4118-92f4-c4d440612db5
# ╟─602e6920-54bd-4579-9b5d-63e477d1431e
# ╟─50982a4f-acae-407a-b7b4-6a1ab653eeec
# ╟─3e6030a0-81e6-43a7-8bc6-beb6ef9aab87
# ╟─0696c9bd-2ecf-4cf8-8881-b76730814342
# ╟─ac006f54-7108-459c-a800-6832b2de254d
# ╟─22ca5cae-7ea7-47a4-b551-dcef2158f4ea
# ╟─80a793f7-9c8c-493c-afcc-4beba2f0bf21
# ╟─25f5ce83-c520-415d-9537-3a483a20e178
# ╟─b2e5c268-8dfa-425e-9abf-b3ec50e9c0c5
# ╟─355d7e19-8e92-4ce0-afbd-854ea3a3f33e
# ╟─7cebb1ad-bddf-41b0-8d50-c8a2dc2bd202
# ╟─64834d5f-cb72-4ab4-81ba-7a3d7eb8d8c0
# ╟─ae351013-0928-4544-b043-54091f17565f
# ╟─087d8ccb-0207-4dad-ac44-d25a4835de4f
# ╟─4b899664-f5b9-4932-a3bb-38f818efb60d
# ╟─01c290ac-d448-47bd-ae74-545c6c4d2c00
# ╟─788e7c1b-d58d-4553-bce6-eaa4cc3085ae
# ╟─1a8ac980-8a88-47f3-ac23-3333945c5720
# ╟─313cc81a-3568-40f5-8b3f-76d0ee36abfa
# ╟─c2efcc3d-b5be-4047-ad7d-e80e9186f7f9
# ╟─3d1c6668-9d2b-46a4-82bf-0dae728dfe8b
# ╟─547e2772-5a5f-4cc9-bec2-a9c67fed8b5a
# ╟─5b363889-b66e-4e10-8e51-78791510de48
# ╟─f268a7cd-c193-45ae-8cee-a4a0710a2a3a
# ╟─8a781eb7-e0db-4286-98e8-70a3beb49952
# ╟─6422fa63-d113-45f5-aee7-8df4836138d5
# ╟─9279bed6-cade-4ae7-9441-81facec7ee1c
# ╟─17a01f79-4791-4c09-82d9-369e5bd58b10
# ╟─7bdccd4e-b9e4-4571-b086-b98619022bf1
# ╟─159948ef-588d-4ecd-80e4-dcd335d5d4e7
# ╠═6490d76d-0b7b-4720-91b2-7e05fbfae0f9
# ╟─3432bede-1139-4ae8-a292-9fe3bffdf9b3
# ╟─2b2f840a-95a5-4882-a2b6-79e12e02d805
# ╟─2d72ed24-21f9-4a4b-9fcf-e20d83a5a1c1
# ╟─9da5feca-40f5-49c7-939f-49c6266970e6
# ╟─58c7aabe-b3ae-4bc8-ba61-356f7e0fa31d
# ╟─9e9c6830-7544-4d81-95c6-34dca751f75f
# ╟─4a5b5744-701e-4051-ac88-b2c3d9715413
# ╟─1c5210c8-7f50-4a6f-a37b-5a6b19b25fc7
# ╟─749ee0b8-7cf5-4c44-9029-08732d959958
# ╟─2f816b4e-ce78-4bc9-9a57-75961e944c78
# ╟─cc6fa932-1e64-4379-b4b1-5d4096fbcb7e
# ╟─0488519b-a1cd-40db-8ee9-d2658b576bd2
# ╟─3f56e844-6b96-45a7-ad01-bc76d7951979
# ╟─42507d15-f0b4-480c-95a6-5cc3187684d7
# ╠═62299cf7-0109-4f26-b140-22421a6e249d
# ╠═0ef53efe-603d-4ca5-936f-996be0caf72a
# ╟─907d88dd-e475-4922-a6fc-1c296721dc78
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
