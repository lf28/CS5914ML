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

# ╔═╡ 9edaaf3c-2973-11ed-193f-d500cbe5d239
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

# ╔═╡ f932c963-31d2-4e0f-ad40-2ca2447f4059
TableOfContents()

# ╔═╡ 3d168eee-aa69-4bd0-b9d7-8a9808e59170
ChooseDisplayMode()

# ╔═╡ 457b21c1-6ea9-4983-b724-fb5cbb69739d
md"""

# CS5914 Machine Learning Algorithms


#### Preliminary: linear algebra

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 57945aa2-b984-407a-89b1-57b0b0a2cd75
md"""

## Today



A quick revision on **Linear algebra** 
* a branch of maths particularly useful for machine learning

\

**Highlight** on some key concepts 
* not a comprehensive review

\

Focus on more **Intuition** and **Geometric interpretations**
* rather than formal theoretical treatment
"""

# ╔═╡ d0db39d8-010b-4858-8918-77bbb171fe19
md"""

## Topics to cover
	
"""

# ╔═╡ 09cfe333-fee5-467e-9000-7729cc1cd072
aside((md"""$(@bind next1 Button("next")) 
$(@bind init1 Button("init"))
	"""))

# ╔═╡ 72583883-5329-46fb-bb78-48105e116c30
begin
	init1
	next_idx = [0];
end;

# ╔═╡ 56f84720-19a0-45ae-b311-efcf13c6a72f
begin
	next1
	topics = ["Vectors & Matrices", "Vector operations: scaling, addition, inner product", "Matrix operations: multiplication, inverse, pseudo inverse", "Linear transform: rotation, projection, bijective transform and inverse"]
	@htl "<ul>$([@htl("""<li>$b</li><br>""") for b in topics[1:min(next_idx[1], length(topics))]])</ul>"
end

# ╔═╡ c8398652-1fcf-4a75-9a62-7631b560803f
let
	next1
	next_idx[1] += 1
end;

# ╔═╡ c8d2815f-efd5-475a-adfc-416d9bb75c1d
md"""

# Vectors
"""

# ╔═╡ dc1bb4ed-bc52-4a48-9c14-8e5b1d99db25
md"""

## Scalar, vector, matrix, tensor

"""

# ╔═╡ 4756d444-a986-4d7d-bb1c-5535a172c83e
html"""<center><img src="https://res.cloudinary.com/practicaldev/image/fetch/s--oTgfo1EL--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/adhiraiyan/DeepLearningWithTF2.0/master/notebooks/figures/fig0201a.png" width = "800"/></center>""" 

# ╔═╡ 80f4f2d5-c686-4221-b5c4-e795ac7443cc
md"[(source)](https://dev.to/mmithrakumar/scalars-vectors-matrices-and-tensors-with-tensorflow-2-0-1f66)"

# ╔═╡ 4fa3850a-58be-4772-ae44-59cd458c7fee
md"""

## Notations


Scalars: normal case letters
* ``x,y,\beta,\gamma``


Vectors: **Bold-face** smaller case
* ``\mathbf{x},\mathbf{y}, \boldsymbol{\beta}``
* ``\mathbf{x}^\top``: row vector

Matrices: **Bold-face** capital case
* ``\mathbf{X},\mathbf{A}, \boldsymbol{\Gamma}``  

Tensors: *sans serif* font
* ``\mathsf{X, A, \Gamma}``

"""

# ╔═╡ 9ca74f71-e1f6-43c7-b08a-7afb7d3dfe0d
md"""
## Vectors


A **vector** is an ordered list of ``n`` *scalars*

```math
\large
\mathbf{a} = \begin{bmatrix} 2 \\ 1\end{bmatrix}, \mathbf{b} =\begin{bmatrix} -1\\2 \end{bmatrix}
```

* **bold face** ``\mathbf{a}, \mathbf{b}`` to denote vectors



##

A vector of length ``n``

```math
\large
\mathbf{x} = \begin{bmatrix}x_1\\ x_2\\ \vdots \\ x_n\end{bmatrix}\;\;\; \;\;\;\;\mathbf{x}^\top =\begin{bmatrix}x_1& x_2& \ldots& x_n\end{bmatrix}
```

* *by default*, all vectors are **column vectors**
* ``\mathbf{x}^\top``: a row vector
  * ``\top`` means transpose
"""

# ╔═╡ 9f3ef777-d8d0-4f0b-8beb-acfe7086b579
md"""

## Vectors in ML context


> **Vectors**: natural choice for **data representation**

**Example**: a user record with three features
* *height, weight* and *age*


"""

# ╔═╡ de7b9c37-67a6-4c60-a916-a835235cc904
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/uservec.svg" width = "300"/></center>""" 

# ╔═╡ 305b094f-c4bf-4a6d-a75b-edea6f18a85a
md"""

## Vectors in ML context 


Vectors can also be used to represent categorical data (textual data)


**Example:**

!!! infor ""
	_Orientation **Week** 2022. **Welcome** to the School of **Computer Science** orientation **week**. Please visit the page on information for current **computer science** students to access the orientation **week** timetable and other resources relevant to your year of **study**_


* assume a dictionary (left) and word count vector (right)

```math

\begin{array}{l}
\texttt{welcome}\\
\texttt{study}\\
\texttt{dog}\\
\texttt{computer}\\
\texttt{science}\\
\texttt{week}\\
\vdots
\end{array}

\;\;\;\;\;\;\;\;

\begin{bmatrix}
1 \\
1\\
0 \\
2 \\
2 \\
3\\
\vdots
\end{bmatrix}

```
* a document becomes a vector

"""

# ╔═╡ 6d4861b6-4a0d-4aac-8f08-e6861e2ecf70
md"""

## Vectors visualisation


**Real number axis** (visualisation of scalars)

* a number, say ``\{\frac{1}{2}, \sqrt{2}, \pi\} \in \mathbb{R}``, are represented as points on the axis



"""

# ╔═╡ d3997aa4-4ddf-4498-9d02-fbb511adc8eb
html"""<center><img src="https://mathworld.wolfram.com/images/eps-svg/RealLine_1201.svg" width = "500"/></center>""" 

# ╔═╡ 539f33bf-d1e8-4dd3-aa26-41f28c993ec5
md"""

##

**Vectors** are generalisation of the same concept
* **points** in higher dimensional Euclidean space ``\mathbb{R}^n``



"""

# ╔═╡ 3ea11e49-9edd-43ac-9908-01d766c331a1
md"""

## 

Vectors in ``\mathbb R^3`` (**three dimensional Euclidean space**)



```math
\large 
\textcolor{red}{\mathbf{b} =\begin{bmatrix} 2 \\ 0\\0\end{bmatrix}},\; 
\textcolor{green}{\mathbf{c} = \begin{bmatrix}3 \\ 2\\0 \end{bmatrix}},\;
\textcolor{blue}{\mathbf{a} = \begin{bmatrix} 1 \\ 1\\4\end{bmatrix}}
```

* ``\mathbf{b,c}`` lives in the ``xy`` -- plane
* ``\mathbf{a}`` "sticks out"
"""

# ╔═╡ 4c1e7871-a931-4781-a200-304a2ef253e1
c = [1,1,4]; d=[2,0,0]; e=[3,2,0];

# ╔═╡ 55325e63-06d0-452d-968e-0554a5ba3050
md"""
## Vector semantics


> **Vector** has
> * a **direction** 
> * a **length** (or norm):  *distance* to ``\mathbf{0} =[0,0]^\top`` (*Pythagorean  theorem*)


The length of a vector:

```math
\large
\underbrace{\Vert \mathbf{x}\Vert}_{\small\text{length}}=\text{dist}(\mathbf{x}, \mathbf{0}) = \sqrt{x_1^2 + x_2^2 +\ldots x_n^2} = \sqrt{\sum_{i=1}^n x_i^2}
```




"""

# ╔═╡ 985bcc5a-a829-47ee-adfb-982547d259a4
md"""
## 
"""

# ╔═╡ 6ecfac7c-1b2e-46ad-bbd6-a61d3b3833b5
md""" Add third vector: $(@bind add_c CheckBox(default=false))"""

# ╔═╡ bd2498df-069b-4ea0-9e44-738142a3080e
a = [2, 1]; b = [-1,2];

# ╔═╡ 5d4e270a-b480-40d1-97ac-d7eaa4700766
let
	gr()
 	plot(xlim=[-2,3], ylim=[-1, 3], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	plot!([a[1],a[1]], [a[2],0], ls=:dash, lc=:gray, lw=1, label="")
	plot!([a[1],0], [a[2],a[2]], ls=:dash, lc=:gray, lw=1, label="")
	plot!([b[1],b[1]], [b[2],0], ls=:dash, lc=:gray, lw=1, label="")
	plot!([b[1],0], [b[2],b[2]], ls=:dash, lc=:gray, lw=1, label="")
	annotate!(a[1],a[2], text(L"\mathbf{a} = %$(a)^\top",:blue, :bottom))
	annotate!(b[1],b[2], text(L"\mathbf{b} = %$(b)^\top",:red, :bottom))
end

# ╔═╡ 7ba65fe7-04d0-451a-a7da-852b04117520
c_ = 2 * a;

# ╔═╡ 995c0421-888a-43d8-aa8a-f5c713fbff5b
let
	gr()
 	plt = plot(xlim = [min(a[1], b[1]) - .5 , max(a[1], b[1]) + .5], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	plot!([a[1],a[1]], [a[2],0], ls=:dash, lc=:gray, lw=1, label="")
	plot!([a[1],0], [a[2],a[2]], ls=:dash, lc=:gray, lw=1, label="")
	plot!([b[1],b[1]], [b[2],0], ls=:dash, lc=:gray, lw=1, label="")
	plot!([b[1],0], [b[2],b[2]], ls=:dash, lc=:gray, lw=1, label="")
	annotate!(a[1],a[2], text(L"\mathbf{a}=%$(a)^\top", :blue, :bottom))
	annotate!([0],[0-0.1], text(L"\mathbf{0}", :black, :right))
	annotate!(b[1],b[2], text(L"\mathbf{b}=%$(b)^\top", :red, :bottom))
	θ = acos(dot(a, [1,0])/ norm(a)) * 180/π

	θb = acos(dot(b, [1,0])/ norm(b)) * 180/π
	annotate!(0.5*b[1], 0.5*b[2], text(L"\Vert \mathbf{b}\Vert = \sqrt{(%$(b[1]))^2+%$(b[2])^2}=\sqrt{%$(sum(a.^2))}", 12, :red, :bottom, rotation = θb ))

	if add_c
		quiver!([0], [0],  quiver=([c_[1]], [c_[2]]), lw=2, alpha=0.4)
		plot!(xlim = [minimum([a[1], b[1], c_[1]]) - .5 , maximum([a[1], b[1], c_[1]]) + .5])
		annotate!(c_[1],c_[2], text(L"\mathbf{c}=%$(c_)^\top",:purple,  :bottom))

		θc = acos(dot(c_, [1,0])/ norm(c_)) * 180/π
		annotate!(0.75*c_[1], 0.75*c_[2], text(L"\Vert c\Vert = \sqrt{%$(c_[1])^2+%$(c_[2])^2}=\sqrt{%$(sum(c_.^2))} = 2\Vert a\Vert ", 10, :purple, :top, rotation = θc ))
		annotate!(0.5*a[1], 0.5*a[2], text(L"\Vert \mathbf{a}\Vert=\sqrt{%$(a[1])^2+%$(a[2])^2}=\sqrt{%$(sum(a.^2))}", 10, :blue, :top, rotation = θ ))
	else

		annotate!(0.5*a[1], 0.5*a[2], text(L"\Vert \mathbf{a}\Vert=\sqrt{%$(a[1])^2+%$(a[2])^2}=\sqrt{%$(sum(a.^2))}", 12, :blue,  :top, rotation = θ ))
	end
	plt
end

# ╔═╡ 75bed86e-922e-41fc-8a17-21edb1daf8b6
md"""
## Special vectors

The **zero** vector

```math
\large
\mathbf{0} =\begin{bmatrix} 0, 0, \ldots, 0\end{bmatrix}^\top
```

* ``\mathbf{0} + \mathbf{a} =\mathbf{a}``

The **one** vector

```math
\large
\mathbf{1} =\begin{bmatrix} 1, 1, \ldots, 1\end{bmatrix}^\top
```


**Standard basis vectors**,  denoted as ``\mathbf{e}_i``, *e.g.* ``\mathbf{e}_i\in \mathbb{R}^3``:


```math
\large
\mathbf{e}_1 = \begin{bmatrix}1 \\0 \\0 \end{bmatrix}, \mathbf{e}_2 = \begin{bmatrix}0 \\1 \\0 \end{bmatrix}, \mathbf{e}_3 = \begin{bmatrix}0 \\0 \\1 \end{bmatrix}

```


"""

# ╔═╡ 23a9febb-a962-45d1-ba00-0a88433587aa
md"""
## Unit vectors


> **Unit vectors**:: *unit length* vector
> * also known as **directional vectors**

"""

# ╔═╡ 9efb80be-c272-41d0-a603-40306921feaa
let

	θs = range(0, 2π, 25)

	vs = [cos.(θs) sin.(θs)]

	
	plt = scatter(vs[:,1], vs[:,2], framestyle=:origin, label=L"\mathbf{v}", xlim=[-1.25, 1.25], ylim=[-1.25, 1.25], ratio=1, markersize=5)
	for i in 1:size(vs)[1]
		quiver!([0], [0], quiver=([vs[i, 1]], [vs[i, 2]]), lw=1.5, lc=:gray)
	end
	plt
end

# ╔═╡ 4a4924fe-dd07-4f71-b532-471639871938
md"""

## Vector operations 



* vectors addition (add vectors together)


* scaling 


* inner product
"""

# ╔═╡ c56ec750-0514-4796-a594-325b930bf4d2
md"""

## Vector operations -- addition


```math
\large
\mathbf{a}+\mathbf{b} = \begin{bmatrix}
           a_1 \\
           a_2 \\
           \vdots\\
           a_d
         \end{bmatrix} +  \begin{bmatrix}
           b_1 \\
           b_2 \\
           \vdots\\
           b_d
         \end{bmatrix} = \begin{bmatrix}
           a_1+b_1 \\
           a_2+b_2 \\
           \vdots\\
           a_d+b_d
         \end{bmatrix}

```


Example: ``\mathbf{b} = [2,0,0]^\top, \mathbf{c} =[3,2,0]^\top``

```math

\mathbf{b}+\mathbf{c}=\begin{bmatrix} 2 \\ 0\\0\end{bmatrix}+ \begin{bmatrix}3 \\ 2\\0 \end{bmatrix} = \begin{bmatrix} 2+3 \\ 0+2\\ 0+0\end{bmatrix} = \begin{bmatrix} 5 \\ 2\\ 0\end{bmatrix}
```

"""

# ╔═╡ 1dd53149-6749-4c86-b63a-eb801a827808
md"""

## Example


**Scalar** addition only moves a point left or right 

* **tip-to-tail** rule
"""

# ╔═╡ 7cb9a28f-b457-4857-aba0-219e06ad6e82
let
	gr()
	ylocations = 0.1 
	plt = plot(ylim = [0., 0.15], xminorticks =1, yticks=false, showaxis=:x, size=(650,150), framestyle=:origin, xticks=([-3:1:4;], ["-3", "-2", "-1", "0", "1", "2", "3", "4"]), xlim = [-2, 4], title="Scalar addition of 1+2")
	δ = 0.1

	sample_data = [1, 2, 3]
	for idx in 1:3
		if idx == 2
			plot!([sample_data[1], sample_data[idx]+ sample_data[1]], 0.2 * (3-idx+1) * [ylocations, ylocations], lc=:gray, arrow= :arrow,  st=:path, label="")
			annotate!([sample_data[idx]], 0.2 * (3-idx+1) *[ylocations], text(L"%$(sample_data[idx])", 10, :bottom))
			scatter!([sample_data[1]], [ 0.2 * (3-idx+1) *[ylocations]], label="", c=:gray)
		else
			plot!([0, sample_data[idx]], 0.2 * (3-idx+1) * [ylocations, ylocations], lc=:gray, arrow= :arrow,  st=:path, label="")
			annotate!([sample_data[idx]], 0.2 * (3-idx+1) *[ylocations], text(L"%$(sample_data[idx])", 10, :bottom))
			scatter!([0], [ 0.2 * idx *[ylocations]], label="", c=:gray)
		end
		
	end

	plt
end

# ╔═╡ 77a18b6d-fcb4-4e1b-ae9b-c57b16c5e6c0
md"""
##

**Vector** addition is the same idea
* parallelogram rule or **tip-to-tail**

```math
\large
\textcolor{blue}{\mathbf{a} = \begin{bmatrix} 2 \\ 1\end{bmatrix}},\; \textcolor{red}{\mathbf{b} =\begin{bmatrix} -1\\2 \end{bmatrix}},\; \textcolor{purple}{\mathbf{a}+\mathbf{b} = \begin{bmatrix} 1 \\ 3\end{bmatrix}}
```

"""

# ╔═╡ 057908d5-7f6b-4976-84f4-eabe8ff88c33
let
	gr()
 	plot(xlim=[-1,4], ylim=[-1, 4.5], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	ab₊ = a +b 
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	quiver!([a[1]], [a[2]],  quiver=([b[1]], [b[2]]), lw=2, lc=2, ls=:dash)
	quiver!([0], [0],  quiver=([ab₊[1]], [ab₊[2]]), lw=2)
	annotate!(a[1],a[2], text(L"\mathbf{a}", :top))
	annotate!(b[1],b[2], text(L"\mathbf{b}", :bottom))
	annotate!(ab₊[1],ab₊[2], text(L"\mathbf{a}+\mathbf{b}", :bottom))
end

# ╔═╡ 94027fb4-cbb0-46d3-abd2-795af9019cd7
md"""

## Vector operations -- scaling



$$\large k\cdot \mathbf{a} = k \cdot \begin{bmatrix}
           a_1 \\
           a_2 \\
           \vdots\\
           a_d
         \end{bmatrix} = \begin{bmatrix}
           k\times a_1 \\
           k\times a_2 \\
           \vdots\\
           k\times a_d
         \end{bmatrix},\;\; k\in \mathbb  R \text{ or a scalar}$$

* *geometrically*, shrinking or streching a vector 
  * ``k>0``, the same direction
  * ``k<0``, opposite direction
  * ``k=0``, shrinks to ``\mathbf{0}``


## Vector operations -- scaling

```math
\large
k\cdot \mathbf{a} = \underbrace{\begin{bmatrix}
           a_1 \\
           a_2 \\
           \vdots\\
           a_d
         \end{bmatrix} + \begin{bmatrix}
           a_1 \\
           a_2 \\
           \vdots\\
         a_d
         \end{bmatrix} + \ldots + \begin{bmatrix}
           a_1 \\
           a_2 \\
           \vdots\\
         a_d
         \end{bmatrix}}_{k}= \begin{bmatrix}
           k\times a_1 \\
           k\times a_2 \\
           \vdots\\
           k\times a_d
         \end{bmatrix}
```



* *arithmetically*, **adding** ``k`` copies of ``\mathbf{a}`` together
"""

# ╔═╡ 7a1d2d83-9820-4b61-91fc-20fa7781e992

md"""

## Vector operations -- scaling

```math
\large
k\cdot \mathbf{a} = \underbrace{\begin{bmatrix}
           a_1 \\
           a_2 \\
           \vdots\\
           a_d
         \end{bmatrix} + \begin{bmatrix}
           a_1 \\
           a_2 \\
           \vdots\\
         a_d
         \end{bmatrix} + \ldots + \begin{bmatrix}
           a_1 \\
           a_2 \\
           \vdots\\
         a_d
         \end{bmatrix}}_{k}= \begin{bmatrix}
           k\times a_1 \\
           k\times a_2 \\
           \vdots\\
           k\times a_d
         \end{bmatrix}
```



* *arithmetically*, **adding** ``k`` copies of ``\mathbf{a}`` together
* **generalisation** of _scalar scaling_ (or multiplication): *e.g* ``3\times 1 = 1+ 1+ 1``
* note that ``k\cdot \mathbf{a} = \mathbf{a}\cdot k``

"""

# ╔═╡ b5465dc2-5c9b-448d-9f82-bcbb8dbfe64b
let
	gr()
	ylocations = 0.1 
	plt = plot(ylim = [0., 0.15], xminorticks =1, yticks=false, showaxis=:x, size=(650,150), framestyle=:origin, xticks=([-3:1:4;], ["-3", "-2", "-1", "0", "1", "2", "3", "4"]), xlim = [-1, 4], title="Scalar scaling: "*L"3 \times 1")
	δ = 0.1

	sample_data = [1, 2, 3]
	for idx in 1:3
		plot!([idx-1, idx], 0.2 * (4-idx+1) * [ylocations, ylocations], lc=:gray, arrow= :arrow,  st=:path, label="")
		annotate!([sample_data[idx]], 0.2 * (4-idx+1) *[ylocations], text(L"1", 10, :bottom))
		scatter!([idx-1], [ 0.2 * (4-idx+1) *[ylocations]], label="", c=:gray)		
	end

	idx = 4
	plot!([0, 3], 0.2 * (4-idx+1) * [ylocations, ylocations], lc=:gray, arrow= :arrow,  st=:path, label="")
	annotate!([3], 0.2 * (4-idx+1) *[ylocations], text(L"3", 13, :left))
	scatter!([0], [ 0.2 * (4-idx+1) *[ylocations]], label="", c=:gray)

	plt
end

# ╔═╡ a54dbf58-d082-440f-bc3d-ceebdfabbda6
md"""

## Example
"""

# ╔═╡ 261ca719-e1e3-4ca8-a685-1bc04e2a3e01
md"
``k_a = `` $(@bind ka Slider(-2.2:0.2:2.3,show_value=true,  default=1)),
``k_b = `` $(@bind kb Slider(-2.2:0.2:2.3,show_value=true,  default=1))
"

# ╔═╡ d4d8448c-1148-4c56-a460-acc6279013ba
let
	gr()
 	plot(xlim=[-4.5,4.5], ylim=[-4.5, 4.5], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	# ab₊ = a +b 
	ka_ = ka *a
	kb_ = kb * b
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	quiver!([0], [0],  quiver=([ka_[1]], [ka_[2]]), lw=2)
	quiver!([0], [0],  quiver=([kb_[1]], [kb_[2]]), lw=2)
	# quiver!([a[1]], [a[2]],  quiver=([b[1]], [b[2]]), lw=2, lc=2, ls=:dash)
	# quiver!([0], [0],  quiver=([ab₊[1]], [ab₊[2]]), lw=2)
	annotate!(a[1],a[2], text(L"\mathbf{a}", :top))
	annotate!(b[1],b[2], text(L"\mathbf{b}", :top))
	annotate!(ka_[1],ka_[2], text(L"%$(ka)\mathbf{a}", :bottom))
	annotate!(kb_[1],kb_[2], text(L"%$(kb)\mathbf{b}", :bottom))
	# annotate!(ab₊[1],ab₊[2], text(L"a+b", :bottom))
end

# ╔═╡ e471923d-5997-4989-afca-bb6d4850fbfd
md"""

## Vector subtraction: ``\mathbf{a}-\mathbf{b}``

A special kind of **_addition_**


```math
\Large
\textcolor{purple}{\mathbf{a}-\mathbf{b}} = \mathbf{a} + (-\mathbf{b})
```

* still parallelogram rule or **tip-to-tail**
"""

# ╔═╡ b7c88966-2e86-4aa6-8fef-efa80ab89a13
let
	gr()
 	plot(xlim=[-1,4], ylim=[-1, 3.5], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	ab₊ = a - b 

	ab₊half = a - .5 * b 
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)

	# quiver!([0], [0],  quiver=([-b[1]], [-b[2]]), lw=2)
	quiver!([a[1]], [a[2]],  quiver=([-b[1]], [-b[2]]), lw=2, lc=2, ls=:dash)
	quiver!([0], [0],  quiver=([ab₊[1]], [ab₊[2]]), lw=2)
	annotate!(a[1],a[2], text(L"\mathbf{a}", :bottom))
	annotate!(b[1],b[2], text(L"\mathbf{b}", :bottom))
	annotate!(ab₊half[1], ab₊half[2], text(L"-\mathbf{b}", :bottom))
	annotate!(ab₊[1],ab₊[2], text(L"\mathbf{a}-\mathbf{b}", :top))
end

# ╔═╡ c9796d01-2538-41a4-a853-41557b9ca414
md"""

## Vector operations -- linear combination


**Linear combination**:  combine **addition** and **scaling** together

```math
\large 
k_1 \mathbf{a} + k_2 \mathbf{b} = k_1\begin{bmatrix}
           a_1\\
           a_2\\
           \vdots\\
           a_n
         \end{bmatrix} + k_2\begin{bmatrix}
           b_1\\
           b_2\\
           \vdots\\
           b_n
         \end{bmatrix}= \begin{bmatrix}
           k_1 a_1 +k_2  b_1\\
           k_1 a_2 +k_2  b_2\\
           \vdots\\
           k_1 a_n+k_2  b_n
         \end{bmatrix}

```


* **_scale_** each vectors first
* then **_add_** the scaled vectors


"""

# ╔═╡ b61b593f-4259-4ecc-bad3-719d8913ce20
md"""
``k_1=``$(@bind k1 Slider(-5:0.5:5, default=1, show_value=true)); 
``k_2=`` $(@bind k2 Slider(-5:0.5:5, default=1, show_value=true)) 
"""

# ╔═╡ 84fe36ac-b441-42e7-a7e5-8aae171038b0
let
	gr()
 	plot(xlim=[-13,13], ylim=[-13, 13], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	ka_ = k1 * a
	kb_ = k2 * b
	ab₊ = ka_ +kb_ 
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	quiver!([0], [0],  quiver=([ka_[1]], [ka_[2]]), lw=2)
	quiver!([0], [0],  quiver=([kb_[1]], [kb_[2]]), lw=2)
	quiver!([0], [0],  quiver=([ab₊[1]], [ab₊[2]]), lw=2)
	annotate!(a[1],a[2], text(L"\mathbf{a}", 10, :top))
	annotate!(b[1],b[2], text(L"\mathbf{b}", 10, :top))
	annotate!(ka_[1],ka_[2], text(L"%$(k1)\mathbf{a}", 10, :bottom))
	annotate!(kb_[1],kb_[2], text(L"%$(k2)\mathbf{b}", 10, :bottom))
	annotate!(ab₊[1],ab₊[2], text(L"%$(k1) \mathbf{a}+%$(k2)\mathbf{b}",15 ,:bottom))
end

# ╔═╡ 5ab7947a-dc64-4820-aeb8-d1190c34d1ca
md"""
## 

**Example**: linear combination of three vectors

"""

# ╔═╡ 0987c773-1b82-4cdd-a491-9d6afaabdf6f
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/vecaddition.svg" width = "600"/></center>""" 

# ╔═╡ 11f27cdd-6cce-4ce6-8e20-7689b8a37bbe
plt1, plt2, plt3=let
	gr()
 	plt1 = plot(xlim=[-.1,5], ylim=[-.1, 3], ratio=1, framestyle=:origin)
	plt2 = plot(xlim=[-.1,5], ylim=[-.1, 3], ratio=1, framestyle=:origin)
	plt3 = plot(xlim=[-.1,5], ylim=[-.1, 3], ratio=1, framestyle=:origin)
	oo = [0,0]
	a = [1,0]
	b = [1,1]
	c = [0, 1]
	# ab₊ = a +b 
	k₁ = 2.5
	k₂ = 1.5
	k₃ = 1
	ka = k₁ * a
	kb = k₂ * b
	kc = k₃ * c
	kab= ka + kb
	kabc = ka + kb + kc
	quiver!(plt1, [0], [0], quiver=([a[1]], [a[2]]), lc=1, lw=2)
	quiver!(plt1, [0], [0],  quiver=([b[1]], [b[2]]), lc=1, lw=2)
	quiver!(plt1, [0], [0],  quiver=([c[1]], [c[2]]), lc=1, lw=2)
	quiver!(plt2, [0], [0],  quiver=([ka[1]], [ka[2]]), lc=1, lw=2)
	quiver!(plt2, [0], [0],  quiver=([kb[1]], [kb[2]]), lc=1, lw=2)
	quiver!(plt3, [0], [0],  quiver=([kc[1]], [kc[2]]), lc=1, lw=2)
	quiver!(plt2, [0], [0],  quiver=([kab[1]], [kab[2]]), lc=2, lw=2)
	quiver!(plt3, [0], [0],  quiver=([kab[1]], [kab[2]]), lc=2, lw=2)
	quiver!(plt2, [ka[1]], [ka[2]],  quiver=([kb[1]], [kb[2]]), lw=2, lc=1, ls=:dash)

	quiver!(plt3, [kab[1]], [kab[2]],  quiver=([kc[1]], [kc[2]]), lw=2, lc=1, ls=:dash)
	quiver!(plt3, [0], [0],  quiver=([kabc[1]], [kabc[2]]), lc=3, lw=2)
	# quiver!([0], [0],  quiver=([ab₊[1]], [ab₊[2]]), lw=2)
	annotate!(plt1, a[1],a[2], text(L"a", :bottom))
	annotate!(plt1, b[1],b[2], text(L"b", :bottom))
	annotate!(plt1, c[1],c[2], text(L"c", :bottom))
	annotate!(plt2, ka[1],ka[2], text(L"%$(k₁)a", :bottom))
	θb = acos(dot(kb, [1,0])/ norm(kb)) * 180/π
	annotate!(plt2, (ka[1]+kab[1])/2,(ka[2]+kab[2])/2, text(L"%$(k₂)b", :top, rotation= θb))
	annotate!(plt3, (kab[1]+kabc[1])/2,(kab[2]+kabc[2])/2, text(L"%$(k₃)c", :left))
	θab = acos(dot(kab, [1,0])/ norm(kab)) * 180/π
	annotate!(plt2, kab[1]/2,kab[2]/2, text(L"%$(k₁) a+ %$(k₂)b", :red, :bottom, rotation=θab))
	annotate!(plt3, kab[1]/2,kab[2]/2, text(L"%$(k₁) a+ %$(k₂)b", :red, :bottom, rotation=θab))
	θabc = acos(dot(kabc, [1,0])/ norm(kabc)) * 180/π
	annotate!(plt3, kabc[1]/2,kabc[2]/2, text(L"%$(k₁) a+ %$(k₂)b + %$(k₃)c", :green, :bottom, rotation=θabc))
	plt1, plt2, plt3
end;

# ╔═╡ 618d0b95-4a2e-48f1-91ca-b2327c84977a
plot(plt1, plt2,  plt3, size=(900, 200), layout=(1,3))

# ╔═╡ 73736e61-86ec-4b72-b38a-3065c6e122e6
md"""
## Span


> **Span** is the set of all possible linear combinations
> 
> $$\text{span}( \{\mathbf{a}_{1}, \mathbf{a}_2, \ldots, \mathbf{a}_m\}) = \left\{\sum_{j=1}^m k_j \mathbf{a}_j ;\;k_j \in \mathbb R\right \}$$

##

**Example**: the **span** of the basis vectors ``\mathbf{e}_1 = \begin{bmatrix} 1\\0\end{bmatrix}, \mathbf{e}_2=\begin{bmatrix} 0\\1\end{bmatrix}`` in ``\mathbb{R}^2``: the set 

$\left \{x \begin{bmatrix}1\\0\end{bmatrix}+ y \begin{bmatrix}0\\1\end{bmatrix}, \text{for all } x,y\in \mathbb R\right \}$ 

that is

$\left \{\begin{bmatrix}x\\y\end{bmatrix}, \text{for all } x,y\in \mathbb R \right \} =\mathbb{R}^2$ 
"""

# ╔═╡ 97b60054-0848-4285-95ff-e16692baf801
plt_span=let
	gr()
 	plot(xlim=[-2.5,2.5], ylim=[-3, 3],  ratio=1, framestyle=:origin)
	plot!(-3:0.1:3, 3.1 * ones(length(-3:.1:3)), fill= (-3, 0.5, :gray), label="")
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	a = [1,0]
	b = [0,1]
	xs = [[-2, 2], [2, 2], [-2, -2], [2, -2]]
	# k1 = -2
	# k2 = 1
	# ka_ = k1 * a
	# kb_ = k2 * b
	# ab₊ = ka_ +kb_ 
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)

	θs = range(0, 2π, 20)
	vs = [cos.(θs) sin.(θs)] * 1.9
	scatter!(vs[:,1], vs[:,2], framestyle=:origin, label=L"[x, y]^\top", ratio=1, c=:gray, markersize=2)
	for i in 1:size(vs)[1]
		quiver!([0], [0], quiver=([vs[i, 1]], [vs[i, 2]]), lw=0.9, lc=:gray)
	end

	annotate!(a[1],a[2], text(L"\mathbf{e}_1", :top))
	annotate!(b[1],b[2], text(L"\mathbf{e}_2",:left))

end;

# ╔═╡ 66cecbcb-dd5f-45d7-8007-f92c3b60c3ee
 Foldable("Span illustration", plt_span)

# ╔═╡ ed21efbc-6553-4be4-8555-021cd009fb96
md"""

## Span: another example

> What is 
> ```math
> \text{span}\left \{\textcolor{green}{\begin{bmatrix}
>	   2\\
>	   0\\
>	   0
>	 \end{bmatrix}}, \textcolor{red}{\begin{bmatrix}
>	  1.5\\
>	  3\\
>	  0
>	 \end{bmatrix}}\right \} ?
> ```








"""

# ╔═╡ a859cd16-b36f-4201-9f3a-8f37c28e9edc
md"Add the span $(@bind add_span CheckBox(default=false))
Use plotly $(@bind use_plotly CheckBox(default=false))
"

# ╔═╡ c57214b9-2bf6-4129-9ac1-fe03bd507304
begin
	a1_ = [2, 0, 0.];  a2_ = [1.5, 3, 0.]
end;

# ╔═╡ 5d3d7bfd-3d6e-4e5c-932f-d0f5e7326737
md"""
## Vector operations: inner product 


$$\large \mathbf{a}^\top \mathbf{b} = \begin{bmatrix}a_1& a_2& \cdots & a_n\end{bmatrix} \begin{bmatrix}
            b_1 \\
            b_2 \\
           \vdots\\
            b_n
         \end{bmatrix} = \sum_{i=1}^n a_ib_i$$


* the innner product is a **scalar**!




**Example**



```math
 \begin{bmatrix}1& 1& 1\end{bmatrix} \cdot \begin{bmatrix}
            1 \\
            2 \\
            3
         \end{bmatrix} = 1\times 1 + 1\times 2 + 1\times 3 = 6
```



## Some inner product identities 

The following are true

$$\large \mathbf{a}^\top \mathbf{b} = \mathbf{b}^\top\mathbf{a}$$



$$\large \mathbf{a}^\top(\mathbf{b}+\mathbf{c}) = \mathbf{a}^\top\mathbf{b}+ \mathbf{a}^\top\mathbf{c}$$


$$\large(k\mathbf{a})^\top\mathbf{b} = \mathbf{a}^\top(k\mathbf{b})= k(\mathbf{a}^\top\mathbf{b})$$


* they all can be proved based on the inner product definition


"""

# ╔═╡ 2940d90e-3bf3-41b2-8c2c-3b484b9897ee
Foldable("", md"
*Proof*:

```math
\mathbf{a}^\top\mathbf{b} =  \sum_{i=1}^n a_ib_i = \sum_{i=1}^n b_ia_i = \mathbf{b}^\top\mathbf{a}
```

**Example**

```math
 \begin{bmatrix}1& 1& 1\end{bmatrix} \cdot \begin{bmatrix}
            1 \\
            2 \\
            3
         \end{bmatrix} =\begin{bmatrix}1& 2& 3\end{bmatrix} \cdot \begin{bmatrix}
            1 \\
            1 \\
            1
         \end{bmatrix}=   6
```


*Proof*:

```math
(k\mathbf{a})^\top\mathbf{b} =  \sum_{i=1}^n (k \cdot a_i)b_i = k \mathbf{a}^\top\mathbf{b}
```

")

# ╔═╡ d0784cec-9e7c-47d8-a36d-906c0206a476
aside(tip(md"Some write inner product ``\mathbf{a}^\top \mathbf{b}``   as 

```math
\langle \mathbf{a}, \mathbf{b}\rangle
```

"))

# ╔═╡ f6fec48f-b7d9-442c-94d5-5fa938e20526
md"""
## A special inner product: ``\mathbf{a}^\top \mathbf{a}``


$$
\large
\begin{align}
 \mathbf{a}^\top \mathbf{a} &= \begin{bmatrix}a_1& a_2 & \ldots& a_n\end{bmatrix}  \begin{bmatrix}
            a_1 \\
            a_2 \\
           \vdots\\
            a_n
         \end{bmatrix} \\
&= \sum_{i=1}^n a_i^2 = \|\mathbf{a}\|^2

\end{align}$$

- generalisation of scalar square: ``a^2=a \cdot a = |a|^2``
- the *squared* **length** of vector ``\mathbf{a}``: *aka* squared ``L_2`` **norm**

"""

# ╔═╡ a7363b66-e04b-40ec-832c-934f5c47745b
let
	gr()
 	plot(xlim=[-1.5,2.5], ylim=[-.25,2.3],ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	annotate!(a[1],a[2], text(L"\mathbf{a}", :top))
	annotate!(b[1],b[2], text(L"\mathbf{b}", :bottom))
	θ = acos(dot(a, [1,0])/ norm(a)) * 180/π
	annotate!(0.5*a[1], 0.5*a[2], text(L"\sqrt{\mathbf{a}^\top\mathbf{a}}", 18, :top, rotation = θ ))
	θb = acos(dot(b, [1,0])/ norm(b)) * 180/π
	annotate!(0.5*b[1], 0.5*b[2], text(L"\sqrt{\mathbf{b}^\top\mathbf{b}}", 18, :bottom, rotation = θb ))
end

# ╔═╡ d2ecbb5a-71b0-4295-945d-ccc6e8acae5b
md"""

## *Unitify* a vector


Sometimes we want to **standardise** a vector's norm to 1, *a.k.a.* unit directional vector:

```math 
\large
\vec{\mathbf{a}} = \frac{\mathbf{a}}{\sqrt{\mathbf{a}^\top\mathbf{a}}} 
``` 

* has a unit norm (or length of 1).

*Proof:* easy to check the (squared) ``L_2`` norm of ``\vec{\mathbf{a}}`` is
```math
\vec{\mathbf{a}}^\top \vec{\mathbf{a}}=\frac{\mathbf{a}^\top}{\sqrt{\mathbf{a}^\top\mathbf{a}}} \frac{\mathbf{a}}{\sqrt{\mathbf{a}^\top\mathbf{a}}} = \frac{1}{(\sqrt{\mathbf{a}^\top\mathbf{a}})^2}\mathbf{a}^\top\mathbf{a}=1
```


"""

# ╔═╡ b5c9faa1-2d34-4856-b52a-4d7b82a32eb1
let
	gr()
 	plt = plot(xlim=[-1.5,2.5], ylim=[-.25,2.3],ratio=1, framestyle=:origin)
	oo = [0,0]
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	annotate!(a[1],a[2], text(L"\mathbf{a}", :top))
	annotate!(b[1],b[2], text(L"\mathbf{b}", :bottom))
	θ = acos(dot(a, [1,0])/ norm(a)) * 180/π

	av = a / sqrt(sum(a.^2))
	bv = b / sqrt(sum(b.^2))
	quiver!([0], [0], quiver=([av[1]], [av[2]]), lw=4, lc=1)
	quiver!([0], [0], quiver=([bv[1]], [bv[2]]), lw=4, lc=2)
	annotate!(av[1], av[2], text(L"\vec{\mathbf{a}}", 16, :black, :top, rotation = θ ))
	θb = acos(dot(b, [0, 1])/ norm(b)) * 180/π
	annotate!(bv[1]-0.05, bv[2]-0.1, text(L"\vec{\mathbf{b}}", 16, :black, :top ))
	# θb = acos(dot(b, [1,0])/ norm(b)) * 180/π
	# annotate!(0.5*b[1], 0.5*b[2], text(L"\sqrt{\mathbf{b}^\top\mathbf{b}}", 18, :dark, :bottom, rotation = θb ))
	plt
end

# ╔═╡ 77af9dac-fae3-4ab2-9d72-2367fcfe22e0
md"""

## Inner product and cosine rule


Due to cosine's rule, inner product is

$$\Large \mathbf{a}^\top \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\|\cos \theta$$

* ``\theta``: angle between the two vectors
* it relates inner product to geometry
"""

# ╔═╡ d1f7eba1-0c09-4dcd-af6b-087699869f31
let
	gr()
 	plot(xlim=[-1,3.5], ylim=[-1, 3.], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	a = [3,0]
	b=[2,2]
	# bp = dot(a,b)/dot(a,a)*a
	quiver!([0], [0], quiver=([3], [0]), lw=2)
	quiver!([0], [0],  quiver=([2], [2]), lw=2)
	# plot!([2,2], [2,0], ls=:dash, label="")
	annotate!(0+0.6, 0+0.3, text(L"\theta=45^\circ", :top))
	annotate!(a[1],a[2], text(L"\mathbf{a}=%$(a)^\top",:blue, :bottom))
	annotate!(b[1],b[2], text(L"\mathbf{b}=%$(b)^\top",:red, :bottom))
	# annotate!(bp[1],bp[2]-0.1, text(L"b_{\texttt{proj}}", :top))
end

# ╔═╡ 41782dde-3f35-4879-aa89-1ebadd4bf8af
Foldable("Example", md"""

**Example**


$\mathbf{a} = [3,0]^\top, \mathbf{b} =[2,2]^\top$

Inner product definition

```math
\mathbf{a}^\top \mathbf{b} = 3\times 2 + 0 \times 2 =6
```

With cosine definition

```math
\mathbf{a}^\top \mathbf{b} =  \sqrt{3^2+0^2} \sqrt{2^2+2^2} \cdot \cos 45^\circ=6
```
""")

# ╔═╡ e53de410-3bad-4e07-b94f-2285c9ed8c61
md"""
## Orthogonal vectors

If ``\large \mathbf{a} \perp \mathbf{b}`` (``\mathbf{a}, \mathbf{b}`` are **orthogonal**)

```math
\Large
\mathbf{a} \perp \mathbf{b} \Leftrightarrow \mathbf{a}^\top \mathbf{b}=0
```


* ``\mathbf{a} \perp \mathbf{b} \Leftrightarrow  \theta=90^\circ\Leftrightarrow\cos\theta =0 \Leftrightarrow \mathbf{a}^\top\mathbf{b} =0.``

"""

# ╔═╡ 974b566e-c0ec-4c57-9b29-90d975ee8edb
md"""

**Examples**

$$\Large [2,1] \begin{bmatrix}-1\\ 2\end{bmatrix}= -2+2=0$$



"""

# ╔═╡ d56b4250-8fdb-4ac3-945d-aad565ca31f2
md"""


## 


> Zero vector is perpendicular to all vectors
>
> $$\Large \mathbf{0} \perp \mathbf{x}$$ 

*Due to the definition*

```math
\large
 \begin{bmatrix} 0&\ldots & 0\end{bmatrix} \begin{bmatrix}x_1\\ \vdots \\x_n\end{bmatrix}= 0
```

"""

# ╔═╡ 28af146b-9eb2-4490-b89e-fd9cd2965d37
md"""

## Projection via inner product



"""

# ╔═╡ a8dafc84-82d4-472b-b3b5-b1e872632ff3
md"Add project vector: $(@bind add_proj_v CheckBox(default= false)),
proj. length $(@bind k_bproj Slider(-1:0.01:1, default = 1/3))"

# ╔═╡ f2dd24b9-55d3-426f-9a86-80212ff60185
md"""


##

*Alternatively,* 

> the projection can also be computed with **inner products**
> ```math
> \large
> \mathbf{b}_{\text{proj}}  = \frac{\mathbf{a}^\top\mathbf{b}}
> {\mathbf{a}^\top\mathbf{a}} \mathbf{a}
>```
> * it projects ``\mathbf{b}`` to ``\mathbf{a}``
"""

# ╔═╡ f9011bcf-d9fd-41dd-a7b5-e86a461ef390
Foldable("Proof", md"

```math
\|\mathbf{b}\| \cos \theta \times \frac{\mathbf{a}}{\|\mathbf{a}\|} = \|\mathbf{a}\|\|\mathbf{b}\| \cos \theta \times \frac{\mathbf{a}}{\|\mathbf{a}\|^2} =\mathbf{a}^\top\mathbf{b} \frac{\mathbf{a}}{\mathbf{a}^\top\mathbf{a}}
```

")

# ╔═╡ 4c4452e7-9888-43ba-aec5-38f3f77bbb39
proj(x::Vector{T}, a::Vector{T}) where T <: Real = dot(a,x)/dot(a,a) * a ; # project vector x to vector a in Julia

# ╔═╡ 43eb56a6-1463-475f-8925-5e89ae3f03e9
begin
	Random.seed!(2345)
	sample_data = sort(randn(8) * 2)
	μ̄ = mean(sample_data)
end;

# ╔═╡ 2d886d1d-362b-4ce1-a820-097eb415d720
md"""
## Sample mean as *Projection*

The sample mean of ``\mathbf{d} = \{d_1, d_2\ldots, d_n\}`` is


```math
\large
\bar{d} = \frac{1}{n} \sum_i d_i
```
* it *compresses* a bunch of number into one scalar

"""

# ╔═╡ 1a536c90-60d5-41ca-a47a-7b2e6b421429
let
	ylocations = 0.05 * ones(length(sample_data))
	plt = plot(ylim = [0., 0.07], xminorticks =5, yticks=false, showaxis=:x, size=(650,120), framestyle=:origin)
	δ = 0.1
	for i in 1:length(sample_data)
		plot!([sample_data[i]], [ylocations[i]], label="", markershape =:circle, markersize=5, markerstrokewidth=1, st=:sticks, c=1, annotations = (sample_data[i], ylocations[i] + 0.01, Plots.text(L"d_{%$i}", :bottom, 13)))
		# annotate!([sample_data[i]].+7*(-1)^i * δ, [ylocations[i]].+ δ, "", 8)
	end
	# vline!([μ̄], lw=2, ls=:dash, label="sample mean", legend=:topleft)
	plot!([μ̄], [ylocations[1]], label="", markershape =:star5, markersize=5, markerstrokewidth=1, st=:sticks, c=2, annotations = (μ̄, ylocations[1] + 0.01, Plots.text(L"\bar{d}", :bottom, 15)))
	# density!(scientist_data, label="")
	plt
end

# ╔═╡ 06c1da37-1570-486d-8734-22b854e0d78d
md"""
## Sample mean as _Projection_

> **_Sample mean_** is actually a **_projection_**
> * data vector ``\mathbf{d}`` projected to the one vector ``\mathbf{1}``


First note that

```math
\large
\mathbf{1}^\top\mathbf{d} = \sum_{i=1}^n d_i
```

"""

# ╔═╡ a6c388ba-7867-4c14-853c-39eed90a3123
aside(tip(md"
Recall the one vector is
```math
\mathbf{1}_{n\times 1} = \begin{bmatrix} 1 \\ 1\\ \vdots\\ 1\end{bmatrix}
```"))

# ╔═╡ c129df02-5846-4072-809f-1e714fc92fd8
md"""
## 

First note that

```math
\large
\mathbf{1}^\top\mathbf{d} = \sum_{i=1}^n d_i
```


And divide by ``\mathbf{1}^\top\mathbf{1} =\sum_{i=1}^n 1^2= n``, we have the sample mean

```math
\large
 \bar{{d}}=\frac{\sum_i d_i}{n} =\frac{\mathbf{1}^\top \mathbf{d}}{\mathbf{1}^\top\mathbf{1}} 
```

"""

# ╔═╡ e22255d5-03fd-4011-bcf8-4f365a943a8e
md"""
##

First note that

```math
\large
\mathbf{1}^\top\mathbf{d} = \sum_{i=1}^n d_i
```


And divide the sum by ``\mathbf{1}^\top\mathbf{1} = n``, we have the sample mean

```math
\large
\frac{\mathbf{1}^\top \mathbf{d}}{\mathbf{1}^\top\mathbf{1}} = \frac{\sum_i d_i}{n} = \bar{{d}}
```



If we multiply vector ``\mathbf{1}`` on both side, we have

```math
\large 
\frac{\mathbf{1}^\top \mathbf{d}}{\mathbf{1}^\top\mathbf{1}}\mathbf{1} =\bar{{d}}\mathbf{1} =\begin{bmatrix} \bar{d} \\\bar{d} \\ \vdots\\\bar{d}\end{bmatrix}
```

* ``\mathbf{d}``'s **projection** on ``\mathbf{1}``! 

* the original vector is **compressed** to this projected vector.


"""

# ╔═╡ d2a433d3-4f1f-4412-ac44-1f22b9808496
aside(tip(md"Recall the definition of projection:

> ```math
> \large
> \mathbf{b}_{\text{proj}}  = \frac{\mathbf{a}^\top\mathbf{b}}
> {\mathbf{a}^\top\mathbf{a}} \mathbf{a}
>```
> * it projects ``\mathbf{b}`` to ``\mathbf{a}``


"))

# ╔═╡ bf2fe5bf-e0b8-4286-939d-e92cfd84929b
md"""

## Demonstration

"""

# ╔═╡ 301ebcf0-a036-4ba9-a698-ee6c90025295
data = [-1.0, 2.0]; 

# ╔═╡ eb536a05-787c-4f7f-99ce-58a26c79a91c
mean(data)

# ╔═╡ 80dadb83-a797-428e-80d3-c959df90fbdb
proj(data, [1.0, 1.0])

# ╔═╡ 7ec90b3c-2f2d-4830-ac89-70f1e6290755
md"""

# Matrix
"""

# ╔═╡ e2f5a2cc-3663-4bd7-b0be-04a6cab5b77b
md"""
## Matrix -- column view


```math
\mathbf{A} = \begin{pmatrix}
a_{11} & \columncolor{lightsalmon}a_{12} & \ldots & a_{1m}\\
a_{21} & a_{22} & \ldots & a_{2m} \\
\vdots & \vdots &\ddots & \vdots \\
a_{n1} & a_{n2} & \ldots & a_{nm}
\end{pmatrix} = \begin{pmatrix}
\vert & \columncolor{lightsalmon} \vert &  & \vert\\
\mathbf{a}_1 & \mathbf{a}_2 & \ldots & \mathbf{a}_{m} \\
\vert & \vert & & \vert 
\end{pmatrix} 
```

* ``n`` rows, ``m`` columns

Also written in a short hand notation as 

```math
\mathbf{A} = (a_{ij})\in \mathbb{R}^{n\times m}\;\; \text{for}\; i= 1,\ldots n;\; j= 1,\ldots, m

```



"""

# ╔═╡ f59e7733-577a-42e1-81ea-84a68770cec5
md"""

##


**Matrix example** in Machine Learning context

* each **_column_** is a **feature vector**

"""

# ╔═╡ e91e43ce-2880-48c5-8b92-a5e84db70442
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/usermat1.svg" width = "400"/></center>""" 

# ╔═╡ c754e94c-f01f-4aed-9ede-78a1ece67e7c
md"""
## Matrix -- row view

A matrix ``\mathbf{A}\in R^{n\times m}`` : a collection of ``n`` **row vectors** 


```math
\mathbf{A} = \begin{pmatrix}
a_{11} & a_{12} & \ldots & a_{1m}\\
\rowcolor{lightsalmon}  a_{21} & a_{22} & \ldots & a_{2m} \\
\vdots & \vdots &\ddots & \vdots \\
a_{n1} & a_{n2} & \ldots & a_{nm}
\end{pmatrix} = \begin{pmatrix}
  \rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{\alpha}_{1}^\top & \rule[.5ex]{2.5ex}{0.5pt} \\
 \rowcolor{lightsalmon}  \rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{\alpha}_{2}^\top  &  \rule[.5ex]{2.5ex}{0.5pt} \\
 & \vdots &  \\
\rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{\alpha}_{n}^\top  &  \rule[.5ex]{2.5ex}{0.5pt} \\
\end{pmatrix} 
```

where for ``i = 1,\ldots, n``
```math
\boldsymbol{\alpha}_i^\top = \begin{bmatrix} a_{i1}& a_{i2} & \ldots & a_{im} \end{bmatrix}
```
"""

# ╔═╡ e19070ca-b48d-4c62-858d-3b6dbe51ee0c
md"""

##


**Matrix example**: user matrix

* each **_row_** is a **user**'s record

"""

# ╔═╡ d94f969c-9107-4f5b-a2f9-199f51b3843a
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/usermat2.svg" width = "400"/></center>""" 

# ╔═╡ 409f36c6-a7c7-4ed0-9bc9-f24edd2a3ea0
md"""

## Matrix shapes


An ``n\times m`` matrix ``\mathbf{A}`` is


* **tall** if ``n > m``: 
```math
\begin{bmatrix}
\cdot & \cdot \\ \cdot & \cdot \\ \vdots & \vdots \\ \cdot & \cdot \\ \cdot & \cdot\end{bmatrix}
```
* **wide** or **fat** if ``n < m``: 

```math
\begin{bmatrix}\cdot & \cdot & \cdots & \cdot & \cdot\\  \cdot & \cdot & \cdots & \cdot & \cdot  \end{bmatrix}
```
* **square** if ``n=m``

```math
\begin{bmatrix}\cdot & \cdot & \cdot \\ \cdot & \cdot & \cdot \\\cdot & \cdot & \cdot  \end{bmatrix}
```

##

**Vectors** are specific kinds of matrix
  - column vector: ``m=1``, ``\mathbb{R}^{n\times 1}``: ``\begin{bmatrix} \cdot \\ \vdots \\ \cdot \end{bmatrix}``
  - row vector: ``n=1``, ``\mathbb{R}^{1\times m}``: ``\begin{bmatrix}\cdot& \cdots & \cdot\end{bmatrix}``


"""

# ╔═╡ b20cfa48-6af6-4225-9864-869061cbb0d7
md"""

## Matrix addition and scaling

Matrix **addition** is in the same vain as vectors


```math
\mathbf{A} +\mathbf{B} = \begin{pmatrix}
a_{11} + b_{11}& a_{12}+b_{12} & \ldots & a_{1m}+b_{1m}\\
a_{21}+b_{21} & a_{22}+b_{22} & \ldots & a_{2m}+b_{2m} \\
\vdots & \vdots &\ddots & \vdots \\
a_{n1}+b_{n1} & a_{n2}+b_{n2} & \ldots & a_{nm}+b_{nm}
\end{pmatrix}
```

Matrix **scaling** is also the same 


```math
k\mathbf{A} = \begin{pmatrix}
k \cdot a_{11} & k\cdot a_{12} & \ldots & k\cdot a_{1m}\\
k \cdot a_{21} & k \cdot a_{22} & \ldots &k \cdot a_{2m} \\
\vdots & \vdots &\ddots & \vdots \\
k \cdot a_{n1} & k \cdot a_{n2} & \ldots & k \cdot a_{nm}
\end{pmatrix} 
```


And the combination of the two operations:

```math
k_1 \mathbf{A} + k_2 \mathbf{B}
```

"""

# ╔═╡ 2f4ad113-fa7b-4bf9-be29-20c52ecc0043
md"""
## Matrix multiplication 


Matrices of **conforming orders** can be multiplied together, *i.e.* ``\mathbf{A}\in \mathbb{R}^{n\times m}`` and ``\mathbf{B}\in \mathbb{R}^{m\times l}``


> ```math
> \Large
> \mathbf{C}_{n \times l} = \mathbf{A}_{n\times m} \mathbf{B}_{m \times l}
> ```
"""

# ╔═╡ 716129ad-b329-4107-9922-9786d2c67504
html"""<center><img src="https://upload.wikimedia.org/wikipedia/commons/e/eb/Matrix_multiplication_diagram_2.svg" width = "400"/></center>""" 

# ╔═╡ 17312d0a-c01a-429f-ac49-7d9753dc79d7
md"[source](https://commons.wikimedia.org/wiki/File:Matrix_multiplication_diagram_2.svg)"

# ╔═╡ 632dffd8-4f69-48ab-bd51-978f5d3eb21f
# md"""

# * the middle dimension ``m`` **must match**

# ```math

# \mathbf{A}_{n \times m} = \begin{pmatrix}
#   \rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{\alpha}_{1}^\top & \rule[.5ex]{2.5ex}{0.5pt} \\
#  & \vdots &  \\
#  \rowcolor{lightgreen}\rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{\alpha}_{i}^\top  &  \rule[.5ex]{2.5ex}{0.5pt} \\
#  & \vdots &  \\
# \rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{\alpha}_{n}^\top  &  \rule[.5ex]{2.5ex}{0.5pt} \\
# \end{pmatrix} ,\;\; 

# \mathbf{B}_{m \times l} = \begin{pmatrix}
# \vert &  & \columncolor{lightsalmon} \vert &  & \vert\\
# \mathbf{b}_1& \ldots & \mathbf{b}_j & \ldots & \mathbf{b}_{l} \\
# \vert&  & \vert & & \vert 
# \end{pmatrix} 
# ```

# ```math

# (\mathbf{AB})_{n \times l} =\begin{pmatrix}\boldsymbol{\alpha}_1^{\top} \mathbf{b}_1 &  \boldsymbol{\alpha}_1^{\top} \mathbf{b}_2 & \ldots & \boldsymbol{\alpha}_1^{\top} \mathbf{b}_l \\

# \boldsymbol{\alpha}_2^{\top} \mathbf{b}_1 &  \boldsymbol{\alpha}_2^{\top} \mathbf{b}_2 & \ldots & \boldsymbol{\alpha}_2^{\top} \mathbf{b}_l \\
# \vdots & \vdots & \ddots & \vdots\\
# \boldsymbol{\alpha}_n^{\top} \mathbf{b}_1 &  \boldsymbol{\alpha}_n^{\top} \mathbf{b}_2 & \ldots & \boldsymbol{\alpha}_n^{\top} \mathbf{b}_l 
# \end{pmatrix}
# ```
# """

# ╔═╡ 0e75dff7-3763-453e-a984-0623942fe7f4
md"""

## Matrix operations: transpose ``\top``


> **Transpose**: ``(\mathbf{A}^\top)_{ij} = (\mathbf{A})_{ji}``; and ``(\mathbf{A}^\top)^\top=\mathbf{A}``


**Example**


```math
\mathbf{a} = \begin{bmatrix} 1\\2\\3 \end{bmatrix}_{3\times 1}\;\;\;\; \mathbf{a}^\top = \begin{bmatrix} 1 & 2 & 3\end{bmatrix}_{1 \times 3}
```


```math
\mathbf{A} = \begin{bmatrix} a & b & c\\ d& e & f \end{bmatrix}_{2\times 3}\;\;\;\; \mathbf{A}^\top = \begin{bmatrix} a & d \\ b & e \\  c & f \end{bmatrix}_{3\times 2}
```


## Matrix operations

> Matrix multiplication is **not** commutative: ``\mathbf{A}\mathbf{B} \neq \mathbf{BA}``


**Example**

Inner product is a scalar

```math
\underbrace{\Large\mathbf{1}^\top}_{1\times 3} \underbrace{\Large\mathbf{1}}_{3\times 1} = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix}\begin{bmatrix} 1\\1\\1 \end{bmatrix}= 3_{1\times 1}
```

Outer product is a matrix

```math
\underbrace{\Large\mathbf{1}}_{3\times 1} \underbrace{\Large\mathbf{1}^\top}_{1\times 3} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}\begin{bmatrix} 1& 1 & 1 \end{bmatrix}=  \begin{bmatrix} 1 & 1 & 1 \\  1& 1 & 1 \\
1& 1 & 1
\end{bmatrix}_{3\times 3}
```

## Matrix operation rules


> Matrix multiplication is **associative**: ``\mathbf{A}\mathbf{B}\mathbf{C} = \mathbf{A}(\mathbf{BC})= (\mathbf{AB})\mathbf{C}``


> Matrix multiplication is **distributive**: 
> ``\mathbf{A}(\mathbf{B}+\mathbf{C}) = \mathbf{A}\mathbf{B}+\mathbf{AC}``,  and ``(\mathbf{A}+\mathbf{B})\mathbf{C} = \mathbf{AC}+\mathbf{BC}``

> Matrix multi+transpose: ``(\mathbf{AB})^\top = \mathbf{B}^\top\mathbf{A}``


All the above can be recursively applied together, *e.g.* 

> ``(\mathbf{ABC})^\top = \mathbf{C}^\top \mathbf{B}^\top \mathbf{A}^\top``

*Proof:*
``(\mathbf{ABC})^\top = ((\mathbf{AB})\mathbf{C})^\top =\mathbf{C}^\top (\mathbf{AB})^\top= \mathbf{C}^\top \mathbf{B}^\top \mathbf{A}^\top``; 
*You will get the same result if associate the other way around:* *i.e.* ``(\mathbf{ABC})^\top = (\mathbf{A}(\mathbf{B}\mathbf{C}))^\top``*, which is left as an exercise.*
"""

# ╔═╡ 8214e966-32c9-4c44-aaf2-25bbacbfb84d
md"""


## Special matrices

> **Diagonal matrix**:
> ```math
> \large
>  \text{diag}(\{1,2,3\}) = \begin{pmatrix} 1 & 0 & 0\\
> 0 & 2 & 0\\
> 0 & 0 & 3 \end{pmatrix}
> ```


and 

```math
\begin{pmatrix} 0 & 0 & 0\\
 0 & 0 & 0\\
 0 & 0 & 0 \end{pmatrix}
``` is also a diagonal matrix

## Special matrices


> **Identity matrix**: a square matrix with ones on its diagonal
> ```math
> \large
> \mathbf{I}_{n} = \begin{pmatrix} 1 & 0 & \ldots & 0 \\
> 0 & 1 & \ldots & 0 \\
> \vdots & \vdots & \ddots & \vdots\\
> 0 & 0 & \ldots & 1 \\
> \end{pmatrix}
> ```
> * ``\mathbf{I}_{m}\mathbf{A} = \mathbf{AI}_{n} = \mathbf{A}`` (you should verify yourself!)
> * serve the same role of ``1``, note that ``1\times x = x\times 1= x``

## Special matrices

> **Symmetric matrix**
>
> $\large\mathbf{A}^\top =\mathbf{A}$
> * all diagonal matrices are symmetric

**Symmetric matrix example**

```math
 \large
\mathbf{S} = \begin{bmatrix} a & b \\ b& e \end{bmatrix}_{2\times 2}\;\;\;\; \mathbf{S}^\top = \begin{bmatrix} a & b \\ b & e \end{bmatrix}_{2\times 2}
```

"""

# ╔═╡ 585f721f-47f0-427f-84e1-3f1072c849f1
md"""

## Matrix vector product ``\mathbf{Av}``: 
### Inner product view
\

```math
\large
\mathbf{A}_{n\times  m} \mathbf{v}_{m\times 1} = \begin{pmatrix}
  \rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{\alpha}_{1}^\top & \rule[.5ex]{2.5ex}{0.5pt} \\
 & \vdots &  \\
 \rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{\alpha}_{i}^\top  &  \rule[.5ex]{2.5ex}{0.5pt} \\
 & \vdots &  \\
\rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{\alpha}_{n}^\top  &  \rule[.5ex]{2.5ex}{0.5pt} \\
\end{pmatrix}\begin{pmatrix}
\vert \\
\mathbf{v} \\
\vert
\end{pmatrix} =\begin{pmatrix}\boldsymbol{\alpha}_1^{\top} \mathbf{v}\\

\boldsymbol{\alpha}_2^{\top} \mathbf{v}  \\
\vdots \\
\boldsymbol{\alpha}_n^{\top} \mathbf{v} 
\end{pmatrix}_{n\times 1}
```

or 


> ```math
> \Large
> \mathbf{A}_{n\times m} \mathbf{v}_{m \times 1} = \mathbf{u}_{n \times 1}
> ```

"""

# ╔═╡ ec049574-9748-4f14-81d0-2881bf1665fe
md"""
## Matrix vector product ``\mathbf{Av}``: 

### Linear combination view

\
Note that ``\mathbf{A}`` is  a collection of ``\large m`` **column vectors**

```math
\large
\mathbf{A}  =\begin{bmatrix}
           a_{11} & a_{12} & \ldots & a_{1m}\\
           a_{21}& a_{22} & \ldots & a_{2m}\\
           \vdots &\vdots  &\ddots &\vdots\\
           a_{n1}& a_{n2} & \ldots & a_{nm}\\
         \end{bmatrix} = \begin{bmatrix}
	\vert & \vert &  & \vert\\
	\mathbf{a}_1 & \mathbf{a}_2 & \ldots & \mathbf{a}_{m} \\
	\vert & \vert & & \vert 
	\end{bmatrix}  

```



## Matrix vector product ``\mathbf{Av}``: 
### Linear combination view
\

Note that ``\mathbf{A}`` is  a collection of ``\large m`` **column vectors**

```math
\large
\mathbf{A}  =\begin{bmatrix}
           a_{11} & a_{12} & \ldots & a_{1m}\\
           a_{21}& a_{22} & \ldots & a_{2m}\\
           \vdots &\vdots  &\ddots &\vdots\\
           a_{n1}& a_{n2} & \ldots & a_{nm}\\
         \end{bmatrix} = \begin{bmatrix}
	\vert & \vert &  & \vert\\
	\mathbf{a}_1 & \mathbf{a}_2 & \ldots & \mathbf{a}_{m} \\
	\vert & \vert & & \vert 
	\end{bmatrix}  

```
and 


```math
\large
\mathbf{v} =\begin{bmatrix}
	v_1\\
	v_2 \\
	\vdots\\
	
	v_m
	\end{bmatrix}
```


!!! important "Matrix vector: linear combo view"	
	```math
	\large
	\begin{align}
	\mathbf{A}\mathbf{v} &=  \begin{bmatrix}
	\vert & \vert &  & \vert\\
	\mathbf{a}_1 & \mathbf{a}_2 & \ldots & \mathbf{a}_{m} \\
	\vert & \vert & & \vert 
	\end{bmatrix}  \begin{bmatrix}
	v_1\\
	v_2 \\
	\vdots\\
	v_m
	\end{bmatrix}  =  v_1\begin{bmatrix}
           \vert\\
           \mathbf{a}_{1}\\
           \vert
         \end{bmatrix} + v_2\begin{bmatrix}
           \vert\\
           \mathbf{a}_{2}\\
           \vert
         \end{bmatrix} + \ldots v_m\begin{bmatrix}
           \vert\\
           \mathbf{a}_{m}\\
           \vert
         \end{bmatrix}\\
	& = \sum_{i=1}^m v_i \mathbf{a}_i
	\end{align}
	```

* ``\mathbf{Av}`` is just a linear combination of the column vectors of ``\mathbf{A}``


"""

# ╔═╡ 14479772-a4d3-4a53-bc1f-8a05f3d96272
md"""

## Example

The _two views_ are **consistent**: they lead to the **same** result

For example, the inner product view 

```math

\begin{bmatrix}
\rowcolor{lightsalmon}1 & 2 \\
\rowcolor{lightgreen}2 & 1 \\
\rowcolor{lightblue}3 & 2
\end{bmatrix}\begin{bmatrix}
a\\
b
\end{bmatrix} = \begin{bmatrix}
\rowcolor{lightsalmon}[1, 2]^\top [a,  b]\\
\cdot \\
\cdot
\end{bmatrix}

```

"""

# ╔═╡ 75b6f1f1-b2e6-45b2-bb29-e7ed6eb13e0a
md"""

## Example

The _two views_ are **consistent**: they lead to the **same** result

For example, the inner product view 

```math

\begin{bmatrix}
\rowcolor{lightsalmon}1 & 2 \\
\rowcolor{lightgreen}2 & 1 \\
\rowcolor{lightblue}3 & 2
\end{bmatrix}\begin{bmatrix}
a\\
b
\end{bmatrix} = \begin{bmatrix}
\rowcolor{lightsalmon} 1a +2b\\
\cdot \\
\cdot
\end{bmatrix}

```

"""

# ╔═╡ af57aab1-5212-4efe-9090-b48cd238c7ce
md"""

## Example

The _two views_ are **consistent**: they lead to the **same** result


For example, the **inner product** view 

```math

\begin{bmatrix}
\rowcolor{lightsalmon}1 & 2 \\
\rowcolor{lightgreen}2 & 1 \\
\rowcolor{lightblue}3 & 2
\end{bmatrix}\begin{bmatrix}
a\\
b
\end{bmatrix} = \begin{bmatrix}
\rowcolor{lightsalmon} 1a + 2b\\
\rowcolor{lightgreen}2a +1b \\
\rowcolor{lightblue} 3 a+2b
\end{bmatrix}

```


The **column** space or **linear combination** view

```math

\begin{bmatrix}
\columncolor{lightsalmon} 
1 & \columncolor{lightgreen} 2 \\
2 & 1 \\
3 & 2
\end{bmatrix}\begin{bmatrix}
a\\
b
\end{bmatrix} = 

```
"""

# ╔═╡ 3a2868a2-5cb6-41c8-8dbe-492864192b13
md"""

## Example

The _two views_ are **consistent**: they lead to the **same** result

For example, the **inner product** view 

```math

\begin{bmatrix}
\rowcolor{lightsalmon}1 & 2 \\
\rowcolor{lightgreen}2 & 1 \\
\rowcolor{lightblue}3 & 2
\end{bmatrix}\begin{bmatrix}
a\\
b
\end{bmatrix} = \begin{bmatrix}
\rowcolor{lightsalmon} 1a + 2b\\
\rowcolor{lightgreen}2a +1b \\
\rowcolor{lightblue} 3 a+2b
\end{bmatrix}

```


The **column** space or **linear combination** view

```math

\begin{bmatrix}
\columncolor{lightsalmon} 
1 & \columncolor{lightgreen} 2 \\
2 & 1 \\
3 & 2
\end{bmatrix}\begin{bmatrix}
a\\
b
\end{bmatrix} = a \begin{bmatrix}
\columncolor{lightsalmon} 
1  \\
2  \\
3 
\end{bmatrix} + b\begin{bmatrix}
\columncolor{lightgreen} 
2  \\
1  \\
2 
\end{bmatrix} = \begin{bmatrix}
\columncolor{lightsalmon}
 1a &+ &\columncolor{lightgreen}2b\\
2a &+ &1b \\
 3 a &+&2b
\end{bmatrix}

```
"""

# ╔═╡ 86962f11-6363-431e-b771-e0c2eea70116
md"""
## Matrix vector product ``\mathbf{Av} \rightarrow \mathbf{u}`` 

> ```math
> \Large
> \mathbf{A}_{n\times m} \mathbf{v}_{m \times 1} = \mathbf{u}_{n \times 1}
> ```
> ```math
> \Large
> T_{\mathbf{A}}(\mathbf{v}_{m \times 1}) = \mathbf{A}_{n\times m}\mathbf{v}_{m \times 1}
> ```

"""

# ╔═╡ dd6d7793-c37c-45e1-b5c5-38a79e80046a
md"""


*  input ``\mathbf{v}\in \mathbb R^m`` and output ``\mathbf{u}\in \mathbb  R^n``
* ``\mathbf{A}``: (linearly) transformation from the input to output


"""

# ╔═╡ dcb65aae-ba91-499b-9e90-866d13e764c5
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/lineartrans.svg" width = "800"/></center>""" 

# ╔═╡ b605180f-3610-4d7d-9d5f-fec61b13544e
# md"""

# ## Linear combo = Matrix product ``\mathbf{Av}`` 


# It is handy to concatenate the set of vectors together to form a matrix

# ```math
# \{\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_n\}
# ```


# ```math

# \mathbf{A} =  \begin{bmatrix}
# \vert & \vert &  & \vert\\
# \mathbf{a}_1 & \mathbf{a}_2 & \ldots & \mathbf{a}_{n} \\
# \vert & \vert & & \vert 
# \end{bmatrix}  =   \begin{bmatrix}
# \vert \\
# \mathbf{a}_{1}  \\
# \vert 
# \end{bmatrix} + \begin{bmatrix}
# \vert \\
# \mathbf{a}_{2}  \\
# \vert 
# \end{bmatrix}+\ldots +  \begin{bmatrix}
# \vert \\
# \mathbf{a}_{n}  \\
# \vert 
# \end{bmatrix}
# ```


# Then **a linear combination**  can be handily written as a matrix times a vector!


# ```math
# \mathbf{A}\mathbf{k} =  \begin{bmatrix}
# \vert & \vert &  & \vert\\
# \mathbf{a}_1 & \mathbf{a}_2 & \ldots & \mathbf{a}_{n} \\
# \vert & \vert & & \vert 
# \end{bmatrix}  \begin{bmatrix}
# k_1\\
# k_2 \\
# \vdots\\

# k_n
# \end{bmatrix}  =   k_1\begin{bmatrix}
# \vert \\
# \mathbf{a}_{1}  \\
# \vert 
# \end{bmatrix} + k_2\begin{bmatrix}
# \vert \\
# \mathbf{a}_{2}  \\
# \vert 
# \end{bmatrix}+\ldots +  k_n\begin{bmatrix}
# \vert \\
# \mathbf{a}_{n}  \\
# \vert 
# \end{bmatrix} = \sum_n k_n \mathbf{a}_n

# ```



# """

# ╔═╡ 9f88c12e-6ef2-40b5-be46-222a8e1658ca
md"""

##

##### Or as a _map_

```math
\Large
T_{\mathbf{A}}: \mathbb{R}^m \rightarrow \mathbb{R}^n
```
"""

# ╔═╡ 1b5e7b7e-1cae-420b-9439-72fdaba5e8c2
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linearmap.svg" width = "500"/></center>""" 

# ╔═╡ 741326a0-4e2a-4ec3-a229-e65d13e99671
md"""

## Some examples of Av: identity


```math
\large \mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1\end{bmatrix} 
```


What is ``\mathbf{Av} \;?`` 



```math
\large
\begin{bmatrix} 
1 & 0 \\
0 & 1
\end{bmatrix} \begin{bmatrix} 
v_1 \\
v_2
\end{bmatrix} =\begin{bmatrix} 1v_1 +0 \\  0 + 1v_2\end{bmatrix} = \begin{bmatrix} v_1 \\  v_2\end{bmatrix}
```

> Identity matrix ``\mathbf{I}``: does nothing
"""

# ╔═╡ 7ca27ccb-c53a-45fb-8db1-0d6d3aa29c09
# A = [1 0; 0 1];

# ╔═╡ de0a3e68-fe44-43d0-a346-dee6c614eb01
md"Apply transform ``\mathbf{A}=\mathbf{I}`` $(@bind transformI CheckBox(false))"

# ╔═╡ 0c23d915-f529-4d56-96db-694f744d943f
let

	A = [1 0; 0 1]
	θs = range(0, 2π, 25)

	vs = [cos.(θs) sin.(θs)]

	plt = scatter(vs[:,1], vs[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], ratio=1, markersize=3, xlabel=L"v_1", ylabel=L"v_2")
	for i in 1:size(vs)[1]
		quiver!([0], [0], quiver=([vs[i, 1]], [vs[i, 2]]), lw=0.8, lc=1)
	end

	if transformI
		data_after = vs * A' 
		for i in 1:size(vs)[1]
			quiver!([0], [0], quiver=([data_after[i, 1]], [data_after[i, 2]]), lw=1, lc=2, alpha=0.5)
		end
		scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=4, alpha=0.25)
	end
	plt
end

# ╔═╡ 85288179-8df2-4a3c-b4ce-8d5918df4579
md"""

## Some examples of Av: streching

How about 
```math
\large
\mathbf{A} = \begin{bmatrix} 1 + \delta & 0 \\ 0 & 1\end{bmatrix} 
```
* ``\delta`` is a  positive constant, say ``0.5``

What does ``\mathbf{Av}`` do?


"""

# ╔═╡ e9d7fa88-6d84-4faa-aef7-b01362ece780
Foldable("", md"
```math

\begin{bmatrix} 
1+\delta & 0 \\
0 & 1
\end{bmatrix} \begin{bmatrix} 
v_1 \\
v_2
\end{bmatrix} =\begin{bmatrix} (1+\delta) v_1 \\ v_2\end{bmatrix}
```

* it stretches horizontally 

")

# ╔═╡ ef98990b-a4e2-4c6f-bd0b-c6caa5bdb694
begin
	δ = 0.5
	A_strech = [1 + δ 0; 0 1]
end;

# ╔═╡ 1956dded-4118-4f93-971d-4c674206ed8a
md"Apply transform ``\mathbf{A}`` $(@bind transformA2 CheckBox(false))"

# ╔═╡ da9cb66a-3008-470a-be09-0c747a7d725a
let

	A = A_strech
	transform = transformA2
	θs = range(0, 2π, 25)

	vs = [cos.(θs) sin.(θs)]

	plt = scatter(vs[:,1], vs[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", xlim=[-1.7, 1.7], ylim=[-1.8, 1.8], ratio=1, markersize=3, xlabel=L"v_1", ylabel=L"v_2")
	for i in 1:size(vs)[1]
		quiver!([0], [0], quiver=([vs[i, 1]], [vs[i, 2]]), lw=0.8, lc=1)
	end
	if transform
		data_after = vs * A' 
		for i in 1:size(vs)[1]
			quiver!([0], [0], quiver=([data_after[i, 1]], [data_after[i, 2]]), lw=1, lc=2, alpha=0.5)
		end
		scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=4, alpha=0.5)
	end
	plt
end

# ╔═╡ 7188cf47-7fdd-4820-9d30-d01778c2435a
md"""

##

> **Horizontal Stretching**
> ```math
> \large
> \mathbf{A} = \begin{bmatrix} 1 +\delta & 0 \\ 0 & 1\end{bmatrix} 
> ```
"""

# ╔═╡ 5641aefd-5376-4b36-8fbd-f5f8aabe18a1
let
	Random.seed!(100)
	A= A_strech
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"v" *" : before", ratio=1, markersize=ms, legend=:outerright)

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"Av" *" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=0.5, linestyle=:dash)
	end
	l = 2
	for i in Iterators.product([-l, l], [-l, l])
		dl = A * [i[1], i[2]]
		minx, maxx = min(min(minx, dl[1]), dl[1]-i[1]), max(max(maxx, dl[1]), dl[1]-i[1])
		miny, maxy = min(min(miny, dl[2]), dl[2]-i[2]), max(max(maxy, dl[2]), dl[2]-i[2])
		quiver!([i[1]], [i[2]], quiver = ([dl[1]- i[1]], [dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
	end
	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ 76092e1c-9058-4490-b23b-38ff5e350f51
md"""

## Some examples of Av: streching

How about 


```math
\large
\mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1+\delta\end{bmatrix} 
```


What does ``\mathbf{Av}`` do?

"""

# ╔═╡ 83363702-6b8f-45ec-8de9-b2eded7279ee
Foldable("", md"

```math

\begin{bmatrix} 
1 & 0 \\
0 & 1+\delta
\end{bmatrix} \begin{bmatrix} 
v_1 \\
v_2
\end{bmatrix} =\begin{bmatrix} v_1 \\ (1+\delta) v_2\end{bmatrix}
```

* stretch vertically

")

# ╔═╡ 477c28c3-af8a-4b19-bdee-a1b1973ae0c0
begin
	δ₂ = 0.5
	A_strech_vert = [1 0; 0 1+δ₂]
end

# ╔═╡ b9a42c1c-5d12-4f3b-b848-0b70d6096136
md"Apply transform ``\mathbf{A}`` $(@bind transformA3 CheckBox(false))"

# ╔═╡ 8fa1932a-49b2-4511-90a5-740de5a0056f
let

	A = A_strech_vert
	transform = transformA3
	θs = range(0, 2π, 25)
	vs = [cos.(θs) sin.(θs)]
	plt = scatter(vs[:,1], vs[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", xlim=[-1.7, 1.7], ylim=[-1.8, 1.8], ratio=1, markersize=3, xlabel=L"v_1", ylabel=L"v_2")
	for i in 1:size(vs)[1]
		quiver!([0], [0], quiver=([vs[i, 1]], [vs[i, 2]]), lw=0.8, lc=1)
	end
	if transform
		data_after = vs * A' 
		for i in 1:size(vs)[1]
			quiver!([0], [0], quiver=([data_after[i, 1]], [data_after[i, 2]]), lw=1, lc=2, alpha=0.5)
		end
		scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=4, alpha=0.5)
	end
	plt
end

# ╔═╡ b7beaa35-a758-4a34-bf2f-e4cc67819aa9
md"""
##

> **Vertical Stretching**
> ```math
> \large
> \mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1+\delta\end{bmatrix} 
> ```

"""

# ╔═╡ 4eaadc80-a8c1-4d35-8240-aabf9bf5f760
let
	Random.seed!(100)
	A = A_strech_vert
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", ratio=1, markersize=ms, legend=:outerright)

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end
	l = 2
	for i in Iterators.product([-l, l], [-l, l])
		dl = A * [i[1], i[2]]
		minx, maxx = min(min(minx, dl[1]), dl[1]-i[1]), max(max(maxx, dl[1]), dl[1]-i[1])
		miny, maxy = min(min(miny, dl[2]), dl[2]-i[2]), max(max(maxy, dl[2]), dl[2]-i[2])
		quiver!([i[1]], [i[2]], quiver = ([dl[1]- i[1]], [dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
	end
	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ 27e28f79-6a46-43db-a9b0-18a2932ff3af
md"""

## Some examples of Av: stretching

How about


```math
\large
\mathbf{A} = \begin{bmatrix} 1+\delta_1 & 0 \\ 0 & 1+\delta_2\end{bmatrix} 
```


What does ``\mathbf{Av}`` do?


"""

# ╔═╡ 3fd23a96-0c7e-444b-87d3-9ea1adc04f9c
Foldable("", md"
```math

\begin{bmatrix} 
1+\delta_1 & 0 \\
0 & 1+\delta_2
\end{bmatrix} \begin{bmatrix} 
v_1 \\
v_2
\end{bmatrix} =\begin{bmatrix} (1+\delta_1) v_1 \\ (1+\delta_2) v_2\end{bmatrix}
```

* stretch in both directions")

# ╔═╡ 6acf5707-ba94-41ce-9723-9d49121a0478
A_strech_both = let
	δ₁ = 0.5; δ₂ = .5
	A4 = [1+δ₁ 0; 0 1+δ₂]
end;

# ╔═╡ 32834e62-f068-4467-867a-62f9e4b5f3ab
md"Apply transform ``\mathbf{A}`` $(@bind transformA4 CheckBox(false))"

# ╔═╡ fe9f4332-4609-4f7a-8fb1-d7f55eb2d055
let

	A = A_strech_both
	transform = transformA4
	θs = range(0, 2π, 25)

	vs = [cos.(θs) sin.(θs)]

	plt = scatter(vs[:,1], vs[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", xlim=[-1.7, 1.7], ylim=[-1.8, 1.8], ratio=1, markersize=3, xlabel=L"v_1", ylabel=L"v_2")
	for i in 1:size(vs)[1]
		quiver!([0], [0], quiver=([vs[i, 1]], [vs[i, 2]]), lw=0.8, lc=1)
	end
	if transform
		data_after = vs * A' 
		for i in 1:size(vs)[1]
			quiver!([0], [0], quiver=([data_after[i, 1]], [data_after[i, 2]]), lw=1, lc=2, alpha=0.5)
		end
		scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=4, alpha=0.5)
	end
	plt
end

# ╔═╡ 6c8f329f-f52f-4dc6-9a4b-d59469d6879a
md"""
##

> **Horizontal & Vertical Stretching**
> ```math
> \large
> \mathbf{A} = \begin{bmatrix} 1+\delta & 0 \\ 0 & 1+\delta\end{bmatrix} 
> ```

"""

# ╔═╡ c1d77f96-1324-4306-b688-3792690e5eaf
let
	Random.seed!(100)
	A = A_strech_both
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", ratio=1, markersize=ms, legend=:outerright)

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end
	l = 2
	for i in Iterators.product([-l, l], [-l, l])
		dl = A * [i[1], i[2]]
		minx, maxx = min(min(minx, dl[1]), dl[1]-i[1]), max(max(maxx, dl[1]), dl[1]-i[1])
		miny, maxy = min(min(miny, dl[2]), dl[2]-i[2]), max(max(maxy, dl[2]), dl[2]-i[2])
		quiver!([i[1]], [i[2]], quiver = ([dl[1]- i[1]], [dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
	end
	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ e96ea9bb-84f4-4f38-8745-2559c0261511
md"""

## Some examples of Av: reflection

How about


```math
\large
\mathbf{A} = \begin{bmatrix} -1 & 0 \\ 0 & 1\end{bmatrix} 
```


What does ``\mathbf{Av}`` do?

"""

# ╔═╡ 1c90c7dc-7d55-4086-b269-daf484e70d26
Foldable("", md"


```math

\begin{bmatrix} 
-1 & 0 \\
0 & 1
\end{bmatrix} \begin{bmatrix} 
v_1 \\
v_2
\end{bmatrix} =\begin{bmatrix} - v_1 \\ v_2\end{bmatrix}
```

* reflects w.r.t ``y``-axis
")

# ╔═╡ e911a197-3d9a-49ae-9333-a58ff6808333
A_reflect = let
	A = [-1 0; 0 1]
end

# ╔═╡ 54685417-ebdc-4aca-9932-04a768ad6954
let
	Random.seed!(100)
	A = A_reflect
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", ratio=1, markersize=ms, legend=:outerright, xlabel=L"v_1", ylabel=L"v_2")

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end
	l = 2
	for i in Iterators.product([-l, l], [-l, l])
		dl = A * [i[1], i[2]]
		minx, maxx = min(min(minx, dl[1]), dl[1]-i[1]), max(max(maxx, dl[1]), dl[1]-i[1])
		miny, maxy = min(min(miny, dl[2]), dl[2]-i[2]), max(max(maxy, dl[2]), dl[2]-i[2])
		quiver!([i[1]], [i[2]], quiver = (0.3*[dl[1]- i[1]], 0.3*[dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
	end
	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ 6bb8f32f-bbcc-4325-89a6-0c2339d25e6f
md"""


##

> Similarly, **reflection w.r.t ``x``-axis**
> ```math
> \large
> \mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & -1\end{bmatrix} 
> ```


"""

# ╔═╡ dcc52699-9d01-4ba2-afed-19f35030e58e
A_reflect_vect = let
	A = [1 0; 0 -1]
end;

# ╔═╡ 307ea6e5-a338-497a-8205-1395cef17a98
let
	Random.seed!(100)
	A = A_reflect_vect
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", ratio=1, markersize=ms, legend=:outerright)

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end
	l = 2
	for i in Iterators.product([-l, l], [-l, l])
		dl = A * [i[1], i[2]]
		minx, maxx = min(min(minx, dl[1]), dl[1]-i[1]), max(max(maxx, dl[1]), dl[1]-i[1])
		miny, maxy = min(min(miny, dl[2]), dl[2]-i[2]), max(max(maxy, dl[2]), dl[2]-i[2])
		quiver!([i[1]], [i[2]], quiver = (0.2*[dl[1]- i[1]], 0.2* [dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
	end
	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ 649896a8-2c39-4d3f-88f5-21c9709aa58b
md"""

## 

> **And reflection w.r.t. the origin ``O``**
>```math
> \large
> \mathbf{A} = \begin{bmatrix} -1 & 0 \\ 0 & -1\end{bmatrix} 
> ```



"""

# ╔═╡ 2f589134-da20-430a-8d45-074136f30a87
A_reflect_origin = let
	A = [-1 0; 0 -1]
end;

# ╔═╡ dec11636-c628-49b2-93cf-4d464f9a4092
let
	Random.seed!(100)
	A = A_reflect_origin
	n = 20
	data = rand(n,2) * (-3) .+ 1.5

	data[n, :] = [-2, 3]
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", ratio=1, markersize=ms, legend=:outerright)

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end
	l = 2
	for i in Iterators.product([-l, l], [-l, l])
		dl = A * [i[1], i[2]]
		minx, maxx = min(min(minx, dl[1]), dl[1]-i[1]), max(max(maxx, dl[1]), dl[1]-i[1])
		miny, maxy = min(min(miny, dl[2]), dl[2]-i[2]), max(max(maxy, dl[2]), dl[2]-i[2])
		quiver!([i[1]], [i[2]], quiver = (0.3 *[dl[1]- i[1]], 0.3*[dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
	end
	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ 0157266b-f826-40f9-8042-780cc684d503
md"""

## Some examples of Av: sheering

How about


```math
\large
\mathbf{A} = \begin{bmatrix} 1 & -\delta \\ 0 & 1\end{bmatrix} 
```


What does ``\mathbf{Av}`` do?

"""

# ╔═╡ 41306a7f-2585-4059-a21e-b1ce85a3d128
Foldable("", md"
```math

\begin{bmatrix} 
1 & -\delta \\
0 & 1
\end{bmatrix} \begin{bmatrix} 
v_1 \\
v_2
\end{bmatrix} =\begin{bmatrix} v_1 - \delta v_2 \\ v_2\end{bmatrix}
```
")

# ╔═╡ 6bffd6b7-5a99-40d5-8dfb-c54d70b954e6
A_sheering = let
	δ = -0.5
	A = [1 δ; 0 1]
end

# ╔═╡ 77f3be56-026c-4f40-9f07-2244c700b32f
md"Apply transform ``\mathbf{A}`` $(@bind transformA8 CheckBox(false))"

# ╔═╡ 41a83336-e937-4662-93bb-c7b1957c542c
let

	A = A_sheering
	transform = transformA8
	θs = range(0, 2π, 25)

	vs = [cos.(θs) sin.(θs)]

	plt = scatter(vs[:,1], vs[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", xlim=[-1.7, 1.7], ylim=[-1.8, 1.8], ratio=1, markersize=3, xlabel=L"v_1", ylabel=L"v_2")
	for i in 1:size(vs)[1]
		quiver!([0], [0], quiver=([vs[i, 1]], [vs[i, 2]]), lw=0.8, lc=1)
	end
	if transform
		data_after = vs * A' 
		for i in 1:size(vs)[1]
			quiver!([0], [0], quiver=([data_after[i, 1]], [data_after[i, 2]]), lw=1, lc=2, alpha=0.5)
		end
		scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=4, alpha=0.5)


		for i in 1:size(vs)[1]
			plot!([vs[i, 1], data_after[i, 1]] , [vs[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
		end
	end


	
	plt
end

# ╔═╡ 04c51e06-3623-4812-875d-c55884dd6e80
md"""

## 

> **Horizontal sheering matrix**
> ```math
> \large
> \mathbf{A} = \begin{bmatrix} 1 & -\delta \\ 0 & 1\end{bmatrix} 
> ```



"""

# ╔═╡ 75ff0340-3549-493a-9674-aad31c1c0465
let
	Random.seed!(100)
	A = A_sheering
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", ratio=1, markersize=ms, legend=:outerright)

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}"*" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end
	l = 2
	for i in Iterators.product([-l, l], [-l, l])
		dl = A * [i[1], i[2]]
		minx, maxx = min(min(minx, dl[1]), dl[1]-i[1]), max(max(maxx, dl[1]), dl[1]-i[1])
		miny, maxy = min(min(miny, dl[2]), dl[2]-i[2]), max(max(maxy, dl[2]), dl[2]-i[2])
		quiver!([i[1]], [i[2]], quiver = ([dl[1]- i[1]], [dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
	end
	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ 15fbd7c8-7f71-4037-acd0-a2169c16279e
md"""

##

> Similarly, **sheering vertically**
> ```math
> \large
> \mathbf{A} = \begin{bmatrix} 1 & 0 \\ \delta & 1\end{bmatrix} 
> ```

"""

# ╔═╡ e59dda27-d8b1-4084-ac8d-1b93539efd6a
A_sheering_vert = let
	δ = 0.5
	A = [1 0; δ 1]
end;

# ╔═╡ a791d098-02f6-44c9-8273-c20f1d688132
let
	Random.seed!(100)
	A = A_sheering_vert
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", ratio=1, markersize=ms, legend=:outerright)

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}"*" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end
	l = 2
	for i in Iterators.product([-l, l], [-l, l])
		dl = A * [i[1], i[2]]
		minx, maxx = min(min(minx, dl[1]), dl[1]-i[1]), max(max(maxx, dl[1]), dl[1]-i[1])
		miny, maxy = min(min(miny, dl[2]), dl[2]-i[2]), max(max(maxy, dl[2]), dl[2]-i[2])
		quiver!([i[1]], [i[2]], quiver = ([dl[1]- i[1]], [dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
	end
	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ 7b6da19c-05b8-4f5d-9bb9-1ebf9c950a9f
md"""

## Some examples of Av: sheering _both_

How about 


```math
\large
\mathbf{A} = \begin{bmatrix} 1 & -\delta \\ \delta & 1\end{bmatrix} 
```


What does ``\mathbf{Av}`` do for small ``\delta``?

```math

\begin{bmatrix} 
1 & -\delta \\
\delta & 1
\end{bmatrix} \begin{bmatrix} 
v_1 \\
v_2
\end{bmatrix} =\begin{bmatrix} v_1 - \delta v_2 \\ \delta v_1 + v_2\end{bmatrix}
```

"""

# ╔═╡ 0692cf89-5259-4a1f-b847-bd65b3e6a7f8
A_sheering_both = let
	δ = .4
	A = [1 -δ; δ 1]
end;

# ╔═╡ 5863e3a4-8258-4457-b64b-92077a85330a
md"
Apply transformation $(begin @bind add_arrow CheckBox(default=true) end) 
"

# ╔═╡ ae2f66f9-9676-4221-9606-f276a0960dec
plt_rotate = let
	Random.seed!(100)
	A = A_sheering_both
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"v" *" : before", ratio=1, markersize=ms, legend=:outerright, title="What "* L"\mathbf{A}" *" does?")

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))


	if add_arrow
			scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"Av"*" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end
			l = 2
		for i in Iterators.product([-l, l], [-l, l])
			dl = A * [i[1], i[2]]
			quiver!([i[1]], [i[2]], quiver = ([dl[1]- i[1]], [dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
		end
		plot!(title="(almost) Rotations!")
	end

	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end;

# ╔═╡ ed962290-f106-4a11-b3f9-81a054490930
plt_rotate

# ╔═╡ 2ac3166a-0b78-49d3-bca3-0055751b5acd
md"""

## Some examples of ``\mathbf{Av}``: Rotation 


When ``\theta`` is small, 


```math
\large
\begin{bmatrix} 1 & -\theta \\ \theta&  1\end{bmatrix} \approx \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta)&  \cos(\theta)\end{bmatrix}\triangleq \mathbf{R} 
```
* ``\mathbf{R}:`` **rotation matrix**

* ``\mathbf{Rv}``: rotates vectors ``\mathbf{v}`` anti-clock wise by ``\theta``
"""

# ╔═╡ 9ee7af3b-7aa5-417f-a97d-b67fc3fc2898
θ = π/4; θ_in_degree= θ * (360/2π);

# ╔═╡ 933a5cea-3e9e-4661-a43c-f6fa2215e852
Rmat(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)];

# ╔═╡ 4a33c6b0-4bb5-4069-99c5-991d89a3cd1b
dots=let
	r = 1
	θ1s = range(0, 2π, 200)
	θs = range(0, 2π, 100)
	dots = [r .* cos.(θ1s) r.* sin.(θ1s)]
	r₁ = 0.4
	θ₁ = π/4
	c₁ = (r+r₁) * [cos(θ₁), sin(θ₁)] 
	θ₂ = 3 * π/4
	c₂ = (r+r₁) * [cos(θ₂), sin(θ₂)] 
	dots2 = c₁' .+ [r₁ .* cos.(θs) r₁ .* sin.(θs)]
	dots3 = c₂' .+ [r₁ .* cos.(θs) r₁ .* sin.(θs)]
	dots = [dots; dots2; dots3]

	dright = [collect(range(0.5 - 0.1, 0.5 +0.1, 10) ) ones(10)* 0.5]
	dright2 = [ones(10)* 0.5 collect(range(0.5 - 0.1, 0.5 +0.1, 10) )]
	dright = [dright; dright2]
	dleft = [collect(range(-0.1,  0.1, 10) ) zeros(10)]
	dleft2 = [zeros(10) collect(range(-0.1, 0.1, 10) )]
	dleft = [-0.5 0.5]  .+ [dleft; dleft2] * Rmat(π/4)'
	dots = [dots; dright; dleft]
end;

# ╔═╡ 03a1a53d-a0be-49e3-a1ca-92c6e7f9d64e
R = Rmat(θ);

# ╔═╡ 06406083-7c5b-479e-9022-64f5b9436304
latexify(R)

# ╔═╡ db9f26fa-9117-4879-beb9-d3315234e2df
let
	Random.seed!(100)
	A = R
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"v" *" : before", ratio=1, markersize=ms, legend=:outerright)

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"Av"*" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end

	if add_arrow
			l = 2
		for i in Iterators.product([-l, l], [-l, l])
			dl = A * [i[1], i[2]]
			quiver!([i[1]], [i[2]], quiver = ([dl[1]- i[1]], [dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
		end
		plot!(title="Rotation: "* L"\theta = %$(round(θ * (360/2π), digits=1))")
	end

	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ 9b63dc70-a96d-4d80-a94a-ae8d7ba70a2f
md"""

##
"""

# ╔═╡ da6c2a51-b782-4e5b-afe8-29ce00583228
md"Rotate me by ``θ``: $(@bind θ_rotate Slider(0:0.05:2π, default= 0, show_value=true))"

# ╔═╡ d8b6d9b8-7d92-4551-82ea-90b9f97958fd
let
	plt = scatter(dots[:, 1], dots[:, 2], xlim = [-1.8, 1.8], label="", ratio =1, framestyle=:origin, markersize=2, alpha=0.3)
	R = Rmat(θ_rotate)
	dots_after = dots * R'
	scatter!(dots_after[:, 1], dots_after[:, 2], label="", markersize=2.5)
	for i in 1:size(dots)[1]
		plot!([dots[i, 1], dots_after[i, 1]] , [dots[i, 2], dots_after[i, 2]], st=:path, label="", lc=2, lw=0.2, linestyle=:dash)
	end
	plt
end

# ╔═╡ 441a05c4-8dd3-4521-8053-8dd67e4115eb
md"""


## Some examples of Av: Projection


Projection is also a linear transformation

> Recall that to project ``\mathbf{b}`` to ``\mathbf{a}``:
> ```math
> \large
> \mathbf{b}_{\text{proj}}  = \frac{\mathbf{a}^\top\mathbf{b}}
> {\mathbf{a}^\top\mathbf{a}}\cdot  \mathbf{a} 
> ```
> ```math
> \large
> \Updownarrow
> ```
> ```math 
> \large
> \mathbf{b}_{\text{proj}}  =  \underbrace{\frac{\mathbf{a} \mathbf{a}^\top}{\mathbf{a}^\top\mathbf{a}}}_{\mathbf{P}_\mathbf{a}}\cdot  \mathbf{b} 
>```


"""

# ╔═╡ 3b3b848e-33b3-4308-b1ba-b8b1067d73d2
Foldable("Details", md"

```math
\frac{\mathbf{a}^\top \mathbf{b}}{\mathbf{a}^\top\mathbf{a}} \cdot \mathbf{a} =  \mathbf{a} \frac{\mathbf{a}^\top \mathbf{b}}{\mathbf{a}^\top\mathbf{a}} =  \frac{\mathbf{a}\mathbf{a}^\top \mathbf{b}}{\mathbf{a}^\top\mathbf{a}}
```

Recall that

```math

k \cdot \mathbf{a} = \mathbf{a}\cdot k
```
")

# ╔═╡ 78d1d2ad-fd48-4b1a-aaa4-812652d960b4
md"""


* note that ``\mathbf{P}_{\mathbf{a}} = \frac{\mathbf{aa}^\top}{\mathbf{a}^\top\mathbf{a}}`` is a ``n\times n`` matrix
* it projects (transforms) ``\mathbf{b}`` to ``\mathbf{a}``

"""

# ╔═╡ d756d2b3-6980-4732-a6fc-241fde7fd8bb
md"""

##

> For example, **project** to the *x*-axis, ``\mathbf{e}_1=[1,0]^\top``, projection matrix is
> ```math
> \large
> \mathbf{P}_{\mathbf{e}_1} = \begin{bmatrix} 1 & 0 \\ 0 & 0\end{bmatrix} 
> ```

"""

# ╔═╡ 7be2099d-5bae-4640-a7d0-9f5d8fecb791
md"""

```math
\mathbf{P}_{\mathbf{e}_1} = \frac{\mathbf{e}_1 \mathbf{e}_1^\top}{\mathbf{e}_1^\top\mathbf{e}_1} = \frac{\begin{bmatrix}1\\ 0\end{bmatrix} \begin{bmatrix}1& 0\end{bmatrix}}{\begin{bmatrix}1& 0\end{bmatrix} \begin{bmatrix}1\\ 0\end{bmatrix}} = \frac{\begin{bmatrix}1& 0 \\0 & 0\end{bmatrix} }{1} = \begin{bmatrix}1& 0 \\0 & 0\end{bmatrix}
```
"""

# ╔═╡ 215484d1-96ad-4ba0-a227-a4f06245389c
project_A(a) = a * a' / dot(a,a) ;

# ╔═╡ 158d8d80-2831-40c5-835d-fb7c237bd104
A_proj_x = let
	A = project_A([1., 0.])
end;

# ╔═╡ 006ad765-e94f-4493-b558-bd858cecbe16
let
	Random.seed!(100)
	A = A_proj_x
	n = 30
	data = rand(n,2) * (-3) .+ 1.5
	ms = 4
	plt = scatter(data[:,1], data[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", ratio=1, markersize=ms, legend=:outerright)

	minx, maxx = minimum([-3, data[:, 1]...]), maximum([3, data[:, 1]...])
	miny, maxy = minimum([-3, data[:,2]...]), maximum([3, data[:, 2]...])
	data_after = data * A' 
	minx, maxx = min(minx, minimum(data_after[:, 1])), max(maxx, maximum(data_after[:, 1]))
	miny, maxy = min(miny, minimum(data_after[:, 2])), max(maxy, maximum(data_after[:, 2]))
	scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}"*" : after",  markersize=ms)
	for i in 1:n
		plot!([data[i, 1], data_after[i, 1]] , [data[i, 2], data_after[i, 2]], st=:path, label="", lc=2, lw=1, linestyle=:dash)
	end
	l = 2
	for i in Iterators.product([-l, l], [-l, l])
		dl = A * [i[1], i[2]]
		minx, maxx = min(min(minx, dl[1]), dl[1]-i[1]), max(max(maxx, dl[1]), dl[1]-i[1])
		miny, maxy = min(min(miny, dl[2]), dl[2]-i[2]), max(max(maxy, dl[2]), dl[2]-i[2])
		quiver!([i[1]], [i[2]], quiver = ([dl[1]- i[1]], [dl[2]- i[2]]), lw=2, lc=:black, linestyle=:dash)
	end
	# xlim_ = [minimum([-3, data[:,1]... data_after[:, 1]...]), maximum([3, data[:, 1]..., data_after[:, 1]...])]
	plot!(xlim=[minx, maxx], ylim=[miny, maxy])

	plt
end

# ╔═╡ 8410e344-7c78-4136-83b5-822dc4862a0d
md"""

## Composition


We can **compose** linear transformations _together_ to form a new transformation

```math
\Large
\mathbf{v} \textcolor{blue}{\xrightarrow{\mathbf{A}_1\cdot}} \textcolor{blue}{\mathbf{v}_1} \textcolor{red}{\xrightarrow{\mathbf{A}_2\cdot }} \textcolor{red}{\mathbf{u}}
```


**For example**

```math
\Large
\mathbf{v} \textcolor{blue}{\xrightarrow{\rm stretch}} \textcolor{blue}{\mathbf{v}_1}  \textcolor{red}{\xrightarrow{\rm rotate}}  \textcolor{red}{\mathbf{u}}
```


> **Mathmetically,** it corresponds to **matrix multiplications** (association rule)
> ```math
> \Large
> \textcolor{red}{\mathbf{u}} = \textcolor{red}{\mathbf{A}_2} \underbrace{\textcolor{blue}{\mathbf{A}_1 }\mathbf{v}}_{\textcolor{blue}{\mathbf{v}_1}}
> ```
"""

# ╔═╡ 22d3a35c-577e-422a-9bbd-9ba48fdba15c
md"""

## Example

"""

# ╔═╡ 40f5c4a2-43fa-475e-983e-6e3a40d8d2c8
A_rotate = Rmat(π/4);

# ╔═╡ c220b509-d7ef-434d-ba66-1d022b72b301
begin
	A1 = A_sheering
	A2 = A_rotate
end;

# ╔═╡ e45c586c-5e7f-459e-960b-6d937c42bd1c
md"""Apply ``\mathbf{A}_1``: $(@bind apply_a1 CheckBox(default=false)), Apply ``\mathbf{A}_2``: $(@bind apply_a2 CheckBox(default=false))"""

# ╔═╡ e749f89d-e47c-4a3d-bfc2-05921de72d3d
let

	A = A1
	transform = apply_a1
	θs = range(0, 2π, 25)

	vs = [cos.(θs) sin.(θs)]

	plt = scatter(vs[:,1], vs[:,2], framestyle=:origin, label=L"\mathbf{v}" *" : before", xlim=[-1.7, 1.7], ylim=[-1.8, 1.8], ratio=1, markersize=3, alpha=0.5, xlabel=L"v_1", ylabel=L"v_2")
	for i in 1:size(vs)[1]
		quiver!([0], [0], quiver=([vs[i, 1]], [vs[i, 2]]), lw=0.8, lc=1)
	end
	if transform
		data_after = vs * A' 
		for i in 1:size(vs)[1]
			quiver!([0], [0], quiver=([data_after[i, 1]], [data_after[i, 2]]), lw=1, lc=2, alpha=0.5)
		end
		scatter!(data_after[:,1], data_after[:,2], markercolor=2, label=L"\mathbf{Av}" *" : after",  markersize=4, alpha=0.5)

		if apply_a2

			data_after = data_after * A2' 
			for i in 1:size(vs)[1]
				quiver!([0], [0], quiver=([data_after[i, 1]], [data_after[i, 2]]), lw=1, lc=3, alpha=0.9)
			end
			scatter!(data_after[:,1], data_after[:,2], markercolor=3, label=L"\mathbf{A}_2\mathbf{A}_1\mathbf{v}" *" : after",  markersize=4, alpha=0.9)


		end
	end
	plt
end

# ╔═╡ a33b1d79-6aef-4334-be4d-268895f0c3c7
md"""

## Composition


We can **compose** more than two linear transformations _together_ 

> **Mathmetically,** it corresponds to multiple **matrix multiplications**
> ```math
> \Large
> \textcolor{red}{\mathbf{u}} = \textcolor{red}{\mathbf{A}_n} \ldots \textcolor{red}{\mathbf{A}_2} \textcolor{red}{\mathbf{A}_1 }\mathbf{v}
> ```
"""

# ╔═╡ efdd86d6-0b19-48d3-8268-ed8ca63274ff
md"""

## Matrix inversion

!!! question "Scalar inverse"
	What is **scalar inverse** 
	
	$\Large a^{-1}$
	* *e.g.* ``5^{-1}`` or ``\pi^{-1}``

> For all ``a\neq 0``, ``a^{-1}\in R`` is defined as another scalar *s.t.*
> ```math 
> \Large
> a a^{-1} =1\;\; \text{or}\;\; a^{-1}a =1
> ```

"""

# ╔═╡ 1313a234-4822-46b0-a7dd-c8c5927a2b85
md"""

## Matrix inverse

What is **scalar inverse** ``a^{-1}``?
> For all ``a\neq 0``, ``a^{-1}\in R`` is defined as a scalar *s.t.*
> ```math 
> \Large
> a a^{-1} =1\;\; \text{or}\;\; a^{-1}a =1
> ```

**Matrix inversion**:

> Given square ``\mathbf{A} \in R^{n\times n}``, its **inversion matrix**, if exists, ``\mathbf{A}^{-1}`` is defined as a ``n\times n`` matrix such that 
> ```math
> \Large
> \mathbf{A}\mathbf{A}^{-1} =\mathbf{I}_{n}\;\; \text{or}\;\; \mathbf{A}^{-1}\mathbf{A}=\mathbf{I}_{n}
> ```


## Matrix inverse

**Not all scalars** has their inverse

* for example ``0``: ``0^{-1}`` **is not defined**


So are **Matrices**
> A lot of matrices **have no inverse!**


> ### But why ?
"""

# ╔═╡ ccfcfd03-b5f3-4df5-97ea-97c7ad5525b5
md"""

## Why some A not invertible?

Consider *Linear* transformation (or just transformation)

```math
\Large
\begin{align}
\mathbf{A}_{n\times m}\mathbf{v} &= \mathbf{u}\\
T: \mathbb{R}^m &\xrightarrow[]{\mathbf{A}} \mathbb{R}^n
\end{align}
```

* input: ``\mathbf{v}\in \mathbb{R}^{m}`` 
* output: ``\mathbf{u}\in \mathbb{R}^n``



!!! important "Invertibility"
	> Invertible ``\mathbf{A}`` ``\Leftrightarrow`` the transformation can be reversed (or cancelled)



"""

# ╔═╡ 7683a628-ca35-4a75-a89f-443afc94db0e
md"""

## Invertible example -- rotation

Rotation matrix 

```math
\large
\mathbf{R} =\begin{bmatrix} \cos\theta & -\sin\theta \\
  \sin\theta & \cos\theta\end{bmatrix} 

```
- it *rotates* $\mathbf{v}$ anti-clockwise by $\theta$, *e.g.* $\theta = \pi/4$

  ``\mathbf{R}=``$(begin latexify_md(Rmat(π/4)) end)

- and rotate **multiple times**: ``\mathbf{R}\mathbf{R} \ldots \mathbf{Rv}``
"""

# ╔═╡ fd0e581f-5d15-4796-8a44-905095509c70
let
	gr()
 	plot(xlim=[-3,3], ylim=[-.5, 3], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	a = [2,1]
	R = Rmat(π/4)
	Ra= R * a
	RRa= R * Ra
	# bp = dot(a,b)/dot(a,a)*a
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([Ra[1]], [Ra[2]]), lw=2)
	quiver!([0], [0],  quiver=([RRa[1]], [RRa[2]]), lw=2)
	# plot!([b[1], bp[1]], [b[2],bp[2]], ls=:dash, lc=:gray, lw=2, label="")
	# annotate!(0+0.3, 0+0.3, text(L"\theta", :top))
	annotate!(a[1],a[2], text(L"\mathbf{v}", :bottom))
	annotate!(Ra[1],Ra[2], text(L"\mathbf{Rv}", :bottom))
	annotate!(RRa[1],RRa[2], text(L"\mathbf{RRv}", :bottom))
	# plot!(Shape([0, aunit[1], abunit[1], bunit[1]], [0, aunit[2], abunit[2], bunit[2]]), lw=1, fillcolor=false)
	# plot!(perp_square([0,0], a, b; δ=0.1), lw=1, fillcolor=false, label="")
	# annotate!(bp[1],bp[2]-0.1, text(L"b_{\texttt{proj}}", :top))
end

# ╔═╡ 8c68e2e5-28c0-4182-944e-5645279f639f
md"""

##

> Is ``\mathbf{R}`` invertible ?


*Intuitively,*  **inverse** ↔ just **rotate back**
* the inversion: ``\mathbf{R}^{-1}`` just *rotate back* clock wise
* in other words, no information is lost in the rotation operation

##

And ``\mathbf{R}^{-1}`` can be found as 

```math
\large
\mathbf{R}^{-1} = \begin{bmatrix} \cos\theta & \sin\theta \\
  -\sin\theta & \cos\theta\end{bmatrix} 

```
* easy to verify ``\mathbf{R}\mathbf{R}^{-1} =\mathbf{I}`` (a composition that cancels each other's effect)


"""

# ╔═╡ c0589e1e-2caa-48d2-b816-daf6f62a93d4
let
	gr()
 	plot(xlim=[-3,3], ylim=[-.5, 3], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	a = [2,1]
	R = Rmat(π/4)
	Ra, RRa= R * a, R * R*a
	# Rinva, RRinva = Rinv*Ra, 
	# bp = dot(a,b)/dot(a,a)*a
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([Ra[1]], [Ra[2]]), lw=2)
	quiver!([0], [0],  quiver=([RRa[1]], [RRa[2]]), lw=2)
	# plot!([b[1], bp[1]], [b[2],bp[2]], ls=:dash, lc=:gray, lw=2, label="")
	# annotate!(0+0.3, 0+0.3, text(L"\theta", :top))
	annotate!(a[1],a[2], text(L"\mathbf{R}^{-1}\mathbf{R}^{-1}\mathbf{u}", :bottom))
	annotate!(Ra[1],Ra[2], text(L"\mathbf{R}^{-1}\mathbf{u}", :bottom))
	annotate!(RRa[1],RRa[2], text(L"\mathbf{u}", :bottom))
	# plot!(Shape([0, aunit[1], abunit[1], bunit[1]], [0, aunit[2], abunit[2], bunit[2]]), lw=1, fillcolor=false)
	# plot!(perp_square([0,0], a, b; δ=0.1), lw=1, fillcolor=false, label="")
	# annotate!(bp[1],bp[2]-0.1, text(L"b_{\texttt{proj}}", :top))

end

# ╔═╡ 79ad9a9f-e25b-4347-a495-271f0d97ea52
md"""

## Non-invertible example -- projection


Projection

```math
\large
\mathbf{b}_{\text{proj}} = \frac{\mathbf{aa}^\top}{\mathbf{a}^\top\mathbf{a}} \mathbf{b} = \mathbf{P}_{a} \mathbf{b}
```


> Is ``\mathbf{P}_{\mathbf{a}}`` invertible?



"""

# ╔═╡ 3c97243d-4c2b-442f-9492-1bb76e31dbba
md"""

## Non-invertible example -- projection


Projection

```math
\large
\mathbf{b}_{\text{proj}} = \frac{\mathbf{aa}^\top}{\mathbf{a}^\top\mathbf{a}} \mathbf{b} = \mathbf{P}_{a} \mathbf{b}
```


> Is ``\mathbf{P}_{\mathbf{a}}`` invertible?

"""

# ╔═╡ 67031a69-f8b4-4aae-9c6e-9f6ccd7c11ad
let
	gr()
 	plot(xlim =[-2.5, 4.9], ylim=[-1.5, 5],  ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	a = [1,1]
	
	data2 = [[4,-1], [3,0], [2,1], [1,2], [0,3], [-1, 4]]
	# b = data
	bp = project_A(a) * data2[1]
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=5, alpha=0.5)
	# quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	# plot!([b[1], bp[1]], [b[2],bp[2]], ls=:dash, lc=:gray, lw=2, label="")
	# plot!(-2:0.5:4.5, x -> -x +3, ls=:dash, lc=:gray, lw=2, label="")
	quiver!([0], [0],  quiver=([bp[1]], [bp[2]]), lw=4, alpha=1.0)
	annotate!(a[1],a[2], text(L"\mathbf{1}", :top))
	# annotate!(b[1],b[2], text(L"\mathbf{d}_1", :top))


	for i in 1:length(data2)
		quiver!([0], [0],  quiver=([data2[i][1]], [data2[i][2]]), lw=2, lc=3)
		annotate!(data2[i][1], data2[i][2], text(L"\mathbf{d}", :left))
	end
	annotate!(bp[1]+0.2,bp[2], text(L"\mathbf{P}_{1}\mathbf{d} =%$(bp)", 10,:left))
	# plot!(perp_square(bp, a, b-bp; δ=0.1), lw=1, label="", fillcolor=false)
end

# ╔═╡ aa5c1733-2975-4eea-aba4-7d45905fe64e
md"""



Projection **NOT invertible**, therefore ``\mathbf{P}_{\mathbf{a}}^{-1}`` **does not** exist

* projection is a many-to-one operation
  * information is lost in the projection operation
* in other words, projection, once done, cannot be reversed!
"""

# ╔═╡ 3d00b8d6-7a40-4f51-b8a9-c4da151dadcd
md"""

## Questions


!!! question "Revision question"
	Explain intuitively why 
	```math
		(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}
	```

	Also prove it by using matrix operations.

	Lastly, why both ``\mathbf{A}``, ``\mathbf{B}`` needs to be invertible to make sure ``\mathbf{AB}`` is invertible?

"""

# ╔═╡ 77b40da6-4577-4968-a48d-f4ba7c6d1bca
md"""

## Appendix
"""

# ╔═╡ 2cdfab17-49c0-4ac4-a199-3ba2e2d5d216
function perp_square(origin, vx, vy; δ=0.1) 
	x = δ * vx/sqrt(norm(vx))
	y = δ * vy/sqrt(norm(vy))
	xyunit = origin+ x + y
	xunit = origin + x
	yunit = origin +y
	Shape([origin[1], xunit[1], xyunit[1], yunit[1]], [origin[2], xunit[2], xyunit[2], yunit[2]])
end

# ╔═╡ aeeda5bf-2146-4c37-957b-63cbb00adb0b
let
	gr()
 	plot(xlim=[-1.5,3], ylim=[-1, 2.5], ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	a = [2,1]
	b= [-1,2]
	# bp = dot(a,b)/dot(a,a)*a
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	# plot!([b[1], bp[1]], [b[2],bp[2]], ls=:dash, lc=:gray, lw=2, label="")
	# annotate!(0+0.3, 0+0.3, text(L"\theta", :top))
	annotate!(a[1],a[2], text(L"\mathbf{a}=%$(a)^\top", :bottom))
	annotate!(b[1],b[2], text(L"\mathbf{b}=%$(b)^\top", :bottom))

	# plot!(Shape([0, aunit[1], abunit[1], bunit[1]], [0, aunit[2], abunit[2], bunit[2]]), lw=1, fillcolor=false)
	plot!(perp_square([0,0], a, b; δ=0.1), lw=1, fillcolor=false, label="")
	# annotate!(bp[1],bp[2]-0.1, text(L"b_{\texttt{proj}}", :top))
end

# ╔═╡ 79ff49e9-4c0d-4e9e-bd6b-080abdf86020
plt_proj_to_a = let
	gr()
 	plt = plot(xlim=[-1,3], ylim=[-1, 3], ratio=1, framestyle=:origin, size=(300, 300))
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	a = [3,0]
	b= [2,2]
	bp = dot(a,b)/dot(a,a)*a
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lc=2, lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lc=1, lw=2)
	
	annotate!(0+0.3, 0+0.3, text(L"\theta", :top))
	annotate!(a[1],a[2], text(L"\mathbf{a}", :bottom))
	annotate!(b[1],b[2], text(L"\mathbf{b}", :bottom))


	if add_proj_v
		plot!([b[1], bp[1]], [b[2],bp[2]], ls=:dash, lc=:gray, lw=2, label="")
		annotate!(bp[1],bp[2]-0.1, text(L"\mathbf{b}_{\texttt{proj}}", :top))
		quiver!([0], [0],  quiver=([bp[1]], [bp[2]]), lc=4, lw=2)
		plot!(perp_square([bp[1],bp[2]], a, b -bp; δ=0.1), lw=1, fillcolor=false, label="")

	else
		quiver!([0], [0],  quiver=([k_bproj* a[1]], [k_bproj* a[2]]), lc=4, lw=3)

	end
	plt
end;

# ╔═╡ 7eb42cc0-fec5-4011-bab5-249defc3ff15
TwoColumn(md"""
\


The projection of ``\mathbf{b}`` on ``\mathbf{a}`` is

\
\

```math
\large
\mathbf{\mathbf{b}}_{\text{proj}} = \overbrace{\|\mathbf{b}\| \cos \theta}^{\rm proj.\; length}\;\, \times \underbrace{\frac{\mathbf{a}}{\|\mathbf{a}\|}}_{\rm unit \; vector}
```

""", plt_proj_to_a)

# ╔═╡ 909974f6-4aa5-482e-934e-64f380314a65
let
	gr()
 	plot( ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	a = [1,1]
	b = data
	bp = dot(a,b)/dot(a,a)*a
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lc=2, lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lc=1, lw=2)
	plot!([b[1], bp[1]], [b[2],bp[2]], ls=:dash, lc=:gray, lw=2, label="")

	quiver!([0], [0],  quiver=([bp[1]], [bp[2]]), ls=:dash, lw=2)
	annotate!(a[1],a[2], text(L"\mathbf{1}", :top))
	annotate!(b[1],b[2], text(L"\mathbf{d}", :bottom))
	# annotate!(bp[1]+0.2,bp[2], text(L"b_{\texttt{proj}} =latexify(:(x = $t))", :left))
	annotate!(bp[1]+0.2,bp[2], text(L"\mathbf{d}_{\texttt{proj}} = \bar{d}\mathbf{1}= %$(bp)^\top", 10,:left))
	plot!(perp_square(bp, a, b-bp; δ=0.1), lw=1, label="", fillcolor=false)
end

# ╔═╡ be040c96-da49-44e6-9a73-7e26a1960261
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

# ╔═╡ 85843ba6-f480-4c0e-a80b-2d5742c0cd72
let
	gr()
 	plt = plot(xlim=[-1,5], ylim=[-1, 5], zlim =[-2,4.5], framestyle=:zerolines, camera=(20,15), size=(600,400), xlabel=L"x", ylabel=L"y", zlabel=L"z")
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	# quiver!([0], [0], [0], quiver=([c[1]], [c[2]], [c[3]]), lw=2)
	# quiver!([0], [0],  quiver=([b[1]], [b[2]]), lw=2)
	arrow3d!([0], [0], [0], [c[1]], [c[2]], [c[3]]; as=0.1, lc=1, la=1, lw=2, scale=:identity)
	# annotate!([c[1]], [c[2]], [c[3]], text("c"))
	arrow3d!([0], [0], [0], [d[1]], [d[2]], [d[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, label="")
	# annotate!([0], [0], [1])
	arrow3d!([0], [0], [0], [e[1]], [e[2]], [e[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	# annotate!([c[1]], [c[2]], [c[3]], text("a"))
	surface!(-1:0.1:5, -1:0.1:5, (x,y) -> 0, colorbar=false, alpha=0.1)
	plt
	# annotate!(a[1],a[2], text(L"a", :top))
	# annotate!(b[1],b[2], text(L"b", :top))
end

# ╔═╡ b912076e-311a-49c3-b324-80d30b7f1baa
let
	if use_plotly
		plotly()
	else
		gr()
	end
	a1 = a1_
	a2 = a2_
	A= hcat([a1, a2]...)
 	plt = plot(xlim=[-4,5], ylim=[-4, 5], zlim =[-1,1.5], framestyle=:zerolines, camera=(20,15), size=(400,400))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, ms =1, label="")
	normv = cross(a1, a2)
	# bv = v₁_ * a1 + v₂_ * a2 
	if add_span
		
			surface!(-4:.2:5, -4:.2:5, (x,y) -> - x * normv[1]/normv[3]- y * normv[2]/normv[3], colorbar=false, alpha=0.5)
		# end
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

[compat]
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.21"
Plots = "~1.38.12"
PlutoTeachingTools = "~0.2.11"
PlutoUI = "~0.7.51"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "d649aec61bf4587fa0ef74e74a7071b4a5d165b3"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

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

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

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

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

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

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

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

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "d014972cd6f5afb1f8cd7adf000b7a966d62c304"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f670f269909a9114df1380cc0fcaa316fff655fb"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.5+0"

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
git-tree-sha1 = "41f7dfb2b20e7e8bf64f6b6fae98f4d2df027b06"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.4"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d03ef538114b38f89d66776f2d8fdc0280f90621"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.12"

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

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

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

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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
# ╟─9edaaf3c-2973-11ed-193f-d500cbe5d239
# ╟─f932c963-31d2-4e0f-ad40-2ca2447f4059
# ╟─3d168eee-aa69-4bd0-b9d7-8a9808e59170
# ╟─457b21c1-6ea9-4983-b724-fb5cbb69739d
# ╟─57945aa2-b984-407a-89b1-57b0b0a2cd75
# ╟─d0db39d8-010b-4858-8918-77bbb171fe19
# ╟─09cfe333-fee5-467e-9000-7729cc1cd072
# ╟─56f84720-19a0-45ae-b311-efcf13c6a72f
# ╟─72583883-5329-46fb-bb78-48105e116c30
# ╟─c8398652-1fcf-4a75-9a62-7631b560803f
# ╟─c8d2815f-efd5-475a-adfc-416d9bb75c1d
# ╟─dc1bb4ed-bc52-4a48-9c14-8e5b1d99db25
# ╟─4756d444-a986-4d7d-bb1c-5535a172c83e
# ╟─80f4f2d5-c686-4221-b5c4-e795ac7443cc
# ╟─4fa3850a-58be-4772-ae44-59cd458c7fee
# ╟─9ca74f71-e1f6-43c7-b08a-7afb7d3dfe0d
# ╟─9f3ef777-d8d0-4f0b-8beb-acfe7086b579
# ╟─de7b9c37-67a6-4c60-a916-a835235cc904
# ╟─305b094f-c4bf-4a6d-a75b-edea6f18a85a
# ╟─6d4861b6-4a0d-4aac-8f08-e6861e2ecf70
# ╟─d3997aa4-4ddf-4498-9d02-fbb511adc8eb
# ╟─539f33bf-d1e8-4dd3-aa26-41f28c993ec5
# ╟─5d4e270a-b480-40d1-97ac-d7eaa4700766
# ╟─3ea11e49-9edd-43ac-9908-01d766c331a1
# ╟─4c1e7871-a931-4781-a200-304a2ef253e1
# ╟─85843ba6-f480-4c0e-a80b-2d5742c0cd72
# ╟─55325e63-06d0-452d-968e-0554a5ba3050
# ╟─985bcc5a-a829-47ee-adfb-982547d259a4
# ╟─6ecfac7c-1b2e-46ad-bbd6-a61d3b3833b5
# ╟─995c0421-888a-43d8-aa8a-f5c713fbff5b
# ╟─bd2498df-069b-4ea0-9e44-738142a3080e
# ╟─7ba65fe7-04d0-451a-a7da-852b04117520
# ╟─75bed86e-922e-41fc-8a17-21edb1daf8b6
# ╟─23a9febb-a962-45d1-ba00-0a88433587aa
# ╟─9efb80be-c272-41d0-a603-40306921feaa
# ╟─4a4924fe-dd07-4f71-b532-471639871938
# ╟─c56ec750-0514-4796-a594-325b930bf4d2
# ╟─1dd53149-6749-4c86-b63a-eb801a827808
# ╟─7cb9a28f-b457-4857-aba0-219e06ad6e82
# ╟─77a18b6d-fcb4-4e1b-ae9b-c57b16c5e6c0
# ╟─057908d5-7f6b-4976-84f4-eabe8ff88c33
# ╟─94027fb4-cbb0-46d3-abd2-795af9019cd7
# ╟─7a1d2d83-9820-4b61-91fc-20fa7781e992
# ╟─b5465dc2-5c9b-448d-9f82-bcbb8dbfe64b
# ╟─a54dbf58-d082-440f-bc3d-ceebdfabbda6
# ╟─261ca719-e1e3-4ca8-a685-1bc04e2a3e01
# ╟─d4d8448c-1148-4c56-a460-acc6279013ba
# ╟─e471923d-5997-4989-afca-bb6d4850fbfd
# ╟─b7c88966-2e86-4aa6-8fef-efa80ab89a13
# ╟─c9796d01-2538-41a4-a853-41557b9ca414
# ╟─b61b593f-4259-4ecc-bad3-719d8913ce20
# ╟─84fe36ac-b441-42e7-a7e5-8aae171038b0
# ╟─5ab7947a-dc64-4820-aeb8-d1190c34d1ca
# ╟─0987c773-1b82-4cdd-a491-9d6afaabdf6f
# ╟─11f27cdd-6cce-4ce6-8e20-7689b8a37bbe
# ╟─618d0b95-4a2e-48f1-91ca-b2327c84977a
# ╟─73736e61-86ec-4b72-b38a-3065c6e122e6
# ╟─66cecbcb-dd5f-45d7-8007-f92c3b60c3ee
# ╟─97b60054-0848-4285-95ff-e16692baf801
# ╟─ed21efbc-6553-4be4-8555-021cd009fb96
# ╟─a859cd16-b36f-4201-9f3a-8f37c28e9edc
# ╠═c57214b9-2bf6-4129-9ac1-fe03bd507304
# ╟─b912076e-311a-49c3-b324-80d30b7f1baa
# ╟─5d3d7bfd-3d6e-4e5c-932f-d0f5e7326737
# ╟─2940d90e-3bf3-41b2-8c2c-3b484b9897ee
# ╟─d0784cec-9e7c-47d8-a36d-906c0206a476
# ╟─f6fec48f-b7d9-442c-94d5-5fa938e20526
# ╟─a7363b66-e04b-40ec-832c-934f5c47745b
# ╟─d2ecbb5a-71b0-4295-945d-ccc6e8acae5b
# ╟─b5c9faa1-2d34-4856-b52a-4d7b82a32eb1
# ╟─77af9dac-fae3-4ab2-9d72-2367fcfe22e0
# ╟─d1f7eba1-0c09-4dcd-af6b-087699869f31
# ╟─41782dde-3f35-4879-aa89-1ebadd4bf8af
# ╟─e53de410-3bad-4e07-b94f-2285c9ed8c61
# ╟─aeeda5bf-2146-4c37-957b-63cbb00adb0b
# ╟─974b566e-c0ec-4c57-9b29-90d975ee8edb
# ╟─d56b4250-8fdb-4ac3-945d-aad565ca31f2
# ╟─28af146b-9eb2-4490-b89e-fd9cd2965d37
# ╟─7eb42cc0-fec5-4011-bab5-249defc3ff15
# ╟─a8dafc84-82d4-472b-b3b5-b1e872632ff3
# ╟─79ff49e9-4c0d-4e9e-bd6b-080abdf86020
# ╟─f2dd24b9-55d3-426f-9a86-80212ff60185
# ╟─f9011bcf-d9fd-41dd-a7b5-e86a461ef390
# ╠═4c4452e7-9888-43ba-aec5-38f3f77bbb39
# ╟─43eb56a6-1463-475f-8925-5e89ae3f03e9
# ╟─2d886d1d-362b-4ce1-a820-097eb415d720
# ╟─1a536c90-60d5-41ca-a47a-7b2e6b421429
# ╟─06c1da37-1570-486d-8734-22b854e0d78d
# ╟─a6c388ba-7867-4c14-853c-39eed90a3123
# ╟─c129df02-5846-4072-809f-1e714fc92fd8
# ╟─e22255d5-03fd-4011-bcf8-4f365a943a8e
# ╟─d2a433d3-4f1f-4412-ac44-1f22b9808496
# ╟─bf2fe5bf-e0b8-4286-939d-e92cfd84929b
# ╠═301ebcf0-a036-4ba9-a698-ee6c90025295
# ╠═eb536a05-787c-4f7f-99ce-58a26c79a91c
# ╠═80dadb83-a797-428e-80d3-c959df90fbdb
# ╟─909974f6-4aa5-482e-934e-64f380314a65
# ╟─7ec90b3c-2f2d-4830-ac89-70f1e6290755
# ╟─e2f5a2cc-3663-4bd7-b0be-04a6cab5b77b
# ╟─f59e7733-577a-42e1-81ea-84a68770cec5
# ╟─e91e43ce-2880-48c5-8b92-a5e84db70442
# ╟─c754e94c-f01f-4aed-9ede-78a1ece67e7c
# ╟─e19070ca-b48d-4c62-858d-3b6dbe51ee0c
# ╟─d94f969c-9107-4f5b-a2f9-199f51b3843a
# ╟─409f36c6-a7c7-4ed0-9bc9-f24edd2a3ea0
# ╟─b20cfa48-6af6-4225-9864-869061cbb0d7
# ╟─2f4ad113-fa7b-4bf9-be29-20c52ecc0043
# ╟─716129ad-b329-4107-9922-9786d2c67504
# ╟─17312d0a-c01a-429f-ac49-7d9753dc79d7
# ╟─632dffd8-4f69-48ab-bd51-978f5d3eb21f
# ╟─0e75dff7-3763-453e-a984-0623942fe7f4
# ╟─8214e966-32c9-4c44-aaf2-25bbacbfb84d
# ╟─585f721f-47f0-427f-84e1-3f1072c849f1
# ╟─ec049574-9748-4f14-81d0-2881bf1665fe
# ╟─14479772-a4d3-4a53-bc1f-8a05f3d96272
# ╟─75b6f1f1-b2e6-45b2-bb29-e7ed6eb13e0a
# ╟─af57aab1-5212-4efe-9090-b48cd238c7ce
# ╟─3a2868a2-5cb6-41c8-8dbe-492864192b13
# ╟─86962f11-6363-431e-b771-e0c2eea70116
# ╟─dd6d7793-c37c-45e1-b5c5-38a79e80046a
# ╟─dcb65aae-ba91-499b-9e90-866d13e764c5
# ╟─b605180f-3610-4d7d-9d5f-fec61b13544e
# ╟─9f88c12e-6ef2-40b5-be46-222a8e1658ca
# ╟─1b5e7b7e-1cae-420b-9439-72fdaba5e8c2
# ╟─741326a0-4e2a-4ec3-a229-e65d13e99671
# ╟─7ca27ccb-c53a-45fb-8db1-0d6d3aa29c09
# ╟─de0a3e68-fe44-43d0-a346-dee6c614eb01
# ╟─0c23d915-f529-4d56-96db-694f744d943f
# ╟─85288179-8df2-4a3c-b4ce-8d5918df4579
# ╟─e9d7fa88-6d84-4faa-aef7-b01362ece780
# ╟─ef98990b-a4e2-4c6f-bd0b-c6caa5bdb694
# ╟─1956dded-4118-4f93-971d-4c674206ed8a
# ╟─da9cb66a-3008-470a-be09-0c747a7d725a
# ╟─7188cf47-7fdd-4820-9d30-d01778c2435a
# ╟─5641aefd-5376-4b36-8fbd-f5f8aabe18a1
# ╟─76092e1c-9058-4490-b23b-38ff5e350f51
# ╟─83363702-6b8f-45ec-8de9-b2eded7279ee
# ╠═477c28c3-af8a-4b19-bdee-a1b1973ae0c0
# ╟─b9a42c1c-5d12-4f3b-b848-0b70d6096136
# ╟─8fa1932a-49b2-4511-90a5-740de5a0056f
# ╟─b7beaa35-a758-4a34-bf2f-e4cc67819aa9
# ╟─4eaadc80-a8c1-4d35-8240-aabf9bf5f760
# ╟─27e28f79-6a46-43db-a9b0-18a2932ff3af
# ╟─3fd23a96-0c7e-444b-87d3-9ea1adc04f9c
# ╠═6acf5707-ba94-41ce-9723-9d49121a0478
# ╟─32834e62-f068-4467-867a-62f9e4b5f3ab
# ╟─fe9f4332-4609-4f7a-8fb1-d7f55eb2d055
# ╟─6c8f329f-f52f-4dc6-9a4b-d59469d6879a
# ╟─c1d77f96-1324-4306-b688-3792690e5eaf
# ╟─e96ea9bb-84f4-4f38-8745-2559c0261511
# ╟─1c90c7dc-7d55-4086-b269-daf484e70d26
# ╠═e911a197-3d9a-49ae-9333-a58ff6808333
# ╟─54685417-ebdc-4aca-9932-04a768ad6954
# ╟─6bb8f32f-bbcc-4325-89a6-0c2339d25e6f
# ╟─dcc52699-9d01-4ba2-afed-19f35030e58e
# ╟─307ea6e5-a338-497a-8205-1395cef17a98
# ╟─649896a8-2c39-4d3f-88f5-21c9709aa58b
# ╟─2f589134-da20-430a-8d45-074136f30a87
# ╟─dec11636-c628-49b2-93cf-4d464f9a4092
# ╟─0157266b-f826-40f9-8042-780cc684d503
# ╟─41306a7f-2585-4059-a21e-b1ce85a3d128
# ╠═6bffd6b7-5a99-40d5-8dfb-c54d70b954e6
# ╟─77f3be56-026c-4f40-9f07-2244c700b32f
# ╟─41a83336-e937-4662-93bb-c7b1957c542c
# ╟─04c51e06-3623-4812-875d-c55884dd6e80
# ╟─75ff0340-3549-493a-9674-aad31c1c0465
# ╟─15fbd7c8-7f71-4037-acd0-a2169c16279e
# ╟─e59dda27-d8b1-4084-ac8d-1b93539efd6a
# ╟─a791d098-02f6-44c9-8273-c20f1d688132
# ╟─4a33c6b0-4bb5-4069-99c5-991d89a3cd1b
# ╟─7b6da19c-05b8-4f5d-9bb9-1ebf9c950a9f
# ╠═0692cf89-5259-4a1f-b847-bd65b3e6a7f8
# ╟─ae2f66f9-9676-4221-9606-f276a0960dec
# ╟─5863e3a4-8258-4457-b64b-92077a85330a
# ╟─ed962290-f106-4a11-b3f9-81a054490930
# ╟─2ac3166a-0b78-49d3-bca3-0055751b5acd
# ╟─9ee7af3b-7aa5-417f-a97d-b67fc3fc2898
# ╟─03a1a53d-a0be-49e3-a1ca-92c6e7f9d64e
# ╟─06406083-7c5b-479e-9022-64f5b9436304
# ╟─933a5cea-3e9e-4661-a43c-f6fa2215e852
# ╟─db9f26fa-9117-4879-beb9-d3315234e2df
# ╟─9b63dc70-a96d-4d80-a94a-ae8d7ba70a2f
# ╟─da6c2a51-b782-4e5b-afe8-29ce00583228
# ╟─d8b6d9b8-7d92-4551-82ea-90b9f97958fd
# ╟─441a05c4-8dd3-4521-8053-8dd67e4115eb
# ╟─3b3b848e-33b3-4308-b1ba-b8b1067d73d2
# ╟─78d1d2ad-fd48-4b1a-aaa4-812652d960b4
# ╟─d756d2b3-6980-4732-a6fc-241fde7fd8bb
# ╟─7be2099d-5bae-4640-a7d0-9f5d8fecb791
# ╟─215484d1-96ad-4ba0-a227-a4f06245389c
# ╟─158d8d80-2831-40c5-835d-fb7c237bd104
# ╟─006ad765-e94f-4493-b558-bd858cecbe16
# ╟─8410e344-7c78-4136-83b5-822dc4862a0d
# ╟─22d3a35c-577e-422a-9bbd-9ba48fdba15c
# ╟─40f5c4a2-43fa-475e-983e-6e3a40d8d2c8
# ╠═c220b509-d7ef-434d-ba66-1d022b72b301
# ╟─e45c586c-5e7f-459e-960b-6d937c42bd1c
# ╟─e749f89d-e47c-4a3d-bfc2-05921de72d3d
# ╟─a33b1d79-6aef-4334-be4d-268895f0c3c7
# ╟─efdd86d6-0b19-48d3-8268-ed8ca63274ff
# ╟─1313a234-4822-46b0-a7dd-c8c5927a2b85
# ╟─ccfcfd03-b5f3-4df5-97ea-97c7ad5525b5
# ╟─7683a628-ca35-4a75-a89f-443afc94db0e
# ╟─fd0e581f-5d15-4796-8a44-905095509c70
# ╟─8c68e2e5-28c0-4182-944e-5645279f639f
# ╟─c0589e1e-2caa-48d2-b816-daf6f62a93d4
# ╟─79ad9a9f-e25b-4347-a495-271f0d97ea52
# ╟─3c97243d-4c2b-442f-9492-1bb76e31dbba
# ╟─67031a69-f8b4-4aae-9c6e-9f6ccd7c11ad
# ╟─aa5c1733-2975-4eea-aba4-7d45905fe64e
# ╟─3d00b8d6-7a40-4f51-b8a9-c4da151dadcd
# ╟─77b40da6-4577-4968-a48d-f4ba7c6d1bca
# ╟─2cdfab17-49c0-4ac4-a199-3ba2e2d5d216
# ╟─be040c96-da49-44e6-9a73-7e26a1960261
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
