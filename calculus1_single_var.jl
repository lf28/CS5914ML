### A Pluto.jl notebook ###
# v0.19.25

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


#### Vector calculus
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 7091d2cf-9237-45b2-b609-f442cd1cdba5
md"""

## Topics to cover
	
"""

# ╔═╡ 0a7f37e1-51bc-427d-a947-31a6be5b765e
aside((md"""$(@bind next1 Button("next")) 
$(@bind init1 Button("init"))
	"""))

# ╔═╡ 595a5ef3-4f54-4502-a943-ace4146efa31
begin
	init1
	next_idx = [0];
end;

# ╔═╡ a696c014-2070-4041-ada3-da79f50c9140
begin
	next1
	topics = ["Single variate calculus: linear & quadratic function, derivative, optimisation", "Multivariate vector calculus: level set, contour, gradient, Hessian", "Local approximation: the essence of differential calculus"]
	@htl "<ul>$([@htl("""<li>$b</li><br>""") for b in topics[1:min(next_idx[1], length(topics))]])</ul>"
end

# ╔═╡ bc1ee08d-9376-44d7-968c-5e114b09a5e0
let
	next1
	next_idx[1] += 1
end;

# ╔═╡ 992a13dd-e6bf-4b18-8654-ac70398e15ab
md"""

# Single variable calculus
"""

# ╔═╡ 49815f5b-f5e4-4cd4-b891-599033fe9d8b
md"""

## Linear function

Univariate linear function ``f: \mathbb{R} \rightarrow \mathbb{R}``

```math
\large
f(x) = b\cdot x+ c
```

* ``c``: intercept
* ``b``: slope
  * *constant* change rate between ``f`` and ``x``
"""

# ╔═╡ 9a2d12cc-59d7-42ef-b1bd-bc90f7c0db3c
md"Add function: $(@bind add_f_linear CheckBox(default=false))"

# ╔═╡ 15428895-7f94-41e4-9fe8-ae2231900afc
md"""
Slope: ``b=`` $(@bind b_ Slider(-10:0.1:10; default = 5, show_value=true)), Intercept: ``c=`` $(@bind c_ Slider(-10:0.1:10; default = 0, show_value=true))
"""

# ╔═╡ 3de289ab-a863-43b3-9799-3ca66791e02c
let
	gr()
	b, c = b_, c_
	plt = plot()
	abs = [(2, 0), (-2, 0)]
	for (a, b) in abs
		plot!(-3:0.1:3, (x) -> a*x+b, framestyle=:origin, label=L"f(x) = %$(a)x + %$(b)", lw =2, lc=:gray,legend=:outerright)
	end

	if add_f_linear
		plot!(-3:0.1:3, (x) -> b*x+c, framestyle=:origin, label="", legend=:outerright, lw=2, r=1)
		x_ = 0 
		if c < 0
			ann_text = text(L"{f(x) = %$(b)x  %$(c)}", :green,  :bottom, rotation = atan(b) * 90/π)
		else
			ann_text = text(L"{f(x) = %$(b)x + %$(c)}",:green,   18, :bottom, rotation = atan(b) * 90/π)
		end
		annotate!([x_], [b*x_ + c], ann_text)
	end
	plt
end

# ╔═╡ 20536535-dd93-4987-886c-5d1d3cccf469
md"""

## Effects of intercept and slope 


### Effect of the intercept: ``c``
\
"""

# ╔═╡ c1a2cdac-605c-4891-abe3-a7e013f390cc
let
	gr()
	b₁, b₀ = 1.5, 0
	plt = plot( legend=:outerright, title="Effect of intercept: "*L"c")

	bbs = [[b₁, b₀]  for b₀ in -3:3]
	for (b₁, b₀) in bbs
		if b₀ < 0 
			anno_text = L"f(x) = %$(b₁)x %$(b₀)"
		else
			anno_text = L"f(x) = %$(b₁)x + %$(b₀)"
		end
		plot!(-1:0.1:3, (x) -> b₁*x+b₀, framestyle=:origin, label=anno_text, legend=:outerright, lw=2)
	end
	plt
end

# ╔═╡ d41c8ca2-a6fa-495e-b8c0-5fe6007f2485
md"""

### Effect of slope: ``b``
"""

# ╔═╡ 3ed1700f-4335-4e2d-b0c6-8a121784e38a
let
	gr()
	a, b = 0, 2
	plt = plot( legend=:outerright, title="Effect of slope: "*L"b")

	abs = [(-2, b),  (-1.5, b), (-1, b), (-0.5, b),  (0,b), (.5, b), (1, b), (1.5, b), (2, b)]
	for (a, b) in abs
		if a == 1.0
			anno_text =	L"f(x) = x + %$(b)"
		else
			anno_text = L"f(x) = %$(a)x + %$(b)"
		end
		plot!(-1:0.1:3, (x) -> a*x+b, framestyle=:origin, label=anno_text, legend=:outerright, lw=2)
	end
	plt
end

# ╔═╡ b4390e22-bf19-4445-8988-954615ac5991
md"""
## Quadratic function

Univariate qudratic function ``\mathbb R\rightarrow \mathbb R``

```math
\large 
f(x) = ax^2 + b x+ c, \;\; a\neq 0

```

> ``a``: quadratic coefficient
* ``a> 0``: bowl facing up
* ``a<0``: bowl facing down
* ``a=0``: reduce to linear function

> ``b``: linear coefficient

> ``c``: the intercept
"""

# ╔═╡ 1dd3214e-de41-4f9d-b43a-35b107c64cf2
md"""

## Quadratic function

"""

# ╔═╡ cf1b8de7-1c04-46a8-8ca6-8c266bc7a6fc
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

# ╔═╡ d2e4fb88-3728-4886-ba4f-634050bbf738
TwoColumn(md"

#### when `` a > 0``


The function has a **minimum**

$(pltapos)
", 
	
	
md" #### when `` a<0``


The function has a **maximum**


$(pltaneg)
")

# ╔═╡ da2f5399-32de-4998-83f2-b84b2d720f82
md"""

## Derivative



$$\large f'(x)= \frac{\mathrm{d}f}{\mathrm{d}x}(x) =\lim_{\Delta x \rightarrow 0} \frac{f(x+\Delta x) - f(x)}{\Delta x}$$


- _instant change rate_ of ``f`` at location ``x``



"""

# ╔═╡ 16ecc090-613e-4746-b12d-0a3d0e4e1727
md"``\Delta x``: $(@bind Δx Slider(1.5:-0.1:0, default=1.5))"

# ╔═╡ 4ef8df63-5b77-48a8-99c0-f014cf6360c1
let
	gr()
	x₀ = 0.0
	xs = -1.2π : 0.1: 1.2π
	f, ∇f = sin, cos
	# anim = @animate for Δx in π:-0.1:0.0
	# Δx = 1.3
	plot(xs, sin, label=L"\sin(x)", ylim = [-1.5, 1.5], xlabel=L"x", lw=2, legend=:outerbottom, framestyle=:semi, title="Derivative at "*L"x=0", legendfontsize=10)
		df = f(x₀ + Δx)-f(x₀)
		k = Δx == 0 ? ∇f(x₀) : df/Δx
		b = f(x₀) - k * x₀ 
		# the approximating linear function with Δx 
		plot!(xs, (x) -> k*x+b, label="", lw=2)
		# the location where the derivative is defined
		scatter!([x₀], [f(x₀)], ms=3, label=L"x_0,\; \sin(x_0)")
		scatter!([x₀+Δx], [f(x₀+Δx)], ms=3, label=L"x_0+Δx,\; \sin(x_0+Δx)")
		plot!([x₀, x₀+Δx], [f(x₀), f(x₀)], lc=:gray, label="")
		plot!([x₀+Δx, x₀+Δx], [f(x₀), f(x₀+Δx)], lc=:gray, label="")
		font_size = Δx < 0.8 ? 12 : 14
		annotate!(x₀+Δx, 0.5 *(f(x₀) + f(x₀+Δx)), text(L"Δf", font_size, :top, rotation = 90))
		annotate!(0.5*(x₀+x₀+Δx), 0, text(L"Δx", font_size,:top))
		annotate!(-.6, 1, text(L"\frac{Δf}{Δx}=%$(round(k, digits=2))", 15,:top))
end

# ╔═╡ e4bd5842-e6af-4e12-af4b-1556b91db0ee
md"""
##

"""

# ╔═╡ 55b1388f-c368-4d95-9893-ea95e8c2359e
let
	gr()
	x₀ = 0.0
	xs = -1.5π : 0.1: 1.5π
	f, ∇f = sin, cos
	anim = @animate for Δx in 1.5:-0.1:0.0
		plot(xs, sin, label=L"\sin(x)", ylim = [-1.5, 1.5], xlabel=L"x", lw=2, legend=:topleft, legendfontsize = 10)
		df = f(x₀ + Δx)-f(x₀)
		k = Δx == 0 ? ∇f(x₀) : df/Δx
		b = f(x₀) - k * x₀ 
		# the approximating linear function with Δx 
		plot!(xs, (x) -> k*x+b, label="", lw=2)
		# the location where the derivative is defined
		scatter!([x₀], [f(x₀)], ms=3, label=L"x_0, \sin(x_0)")
		scatter!([x₀+Δx], [f(x₀+Δx)], ms=3, label=L"x_0+Δx, \sin(x_0+Δx)")
		plot!([x₀, x₀+Δx], [f(x₀), f(x₀)], lc=:gray, label="")
		plot!([x₀+Δx, x₀+Δx], [f(x₀), f(x₀+Δx)], lc=:gray, label="")
		font_size = Δx < 0.8 ? 7 : 10
		annotate!(x₀+Δx, 0.5 *(f(x₀) + f(x₀+Δx)), text(L"Δf=%$(round(df, digits=1))", font_size, :top, rotation = 90))
		annotate!(0.5*(x₀+x₀+Δx), 0, text(L"Δx=%$(round(Δx, digits=1))", font_size,:top))
		annotate!(0, 1, text(L"\frac{Δf}{Δx}=%$(round(k, digits=2))", 10,:top))
	end

	gif(anim, fps=5)
end

# ╔═╡ 6fc139a6-a8b6-4215-a4c6-06a54c2985ec
md"""

## Derivative



$$\large f'(x)= \frac{\mathrm{d}f}{\mathrm{d}x}(x) =\lim_{\Delta x \rightarrow 0} \frac{f(x+\Delta x) - f(x)}{\Delta x}$$

- *also note*, the derivative itself ``f'`` is a ``\mathbb R \rightarrow \mathbb R``  **function**
  - input: ``x\in \mathbb R``
  - outputs: the change rate at $x$ (or the *slope* of the tangent line)



"""

# ╔═╡ a79f418f-d054-46e7-bb9b-f13a7631e21b
html"""<center><img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tangent_function_animation.gif" width = "350"/></center>""" 

# ╔═╡ 8ae96192-2b4f-48e7-b55d-bea27d565671
aside(tip(md"""
Differentiation rules
* constant rule: ``f(x)=c``, then ``f'= 0``
* scalar rule: ``(af(x))' = a f'(x)``
* sum/subtraction rule: ``(f(x)\pm g(x))'= f(x)' \pm g(x)'``
* product rule: ``(f(x)g(x))' = f' g + f g'``
* quotient rule: ``\left (\frac{f(x)}{g(x)}\right )' = \frac{f' g-g'f}{g^2}``
* chain rule: ``h(x) = f(g(x))``, then ``h'(x)=f'(g(x)) g'(x)``"""))

# ╔═╡ bbed688d-0caf-4f93-a19c-ea3a0039a1a2
md"""

## Calculate derivative




Some common derivatives are

```math
f(x)=c, \;\; f'(x) = 0
```

```math
f(x)=bx, \;\; f'(x) = b
```


```math
f(x)=ax^2+bx+c, \;\; f'(x) = 2ax + b
```

```math
f(x)=\exp(x), \;\; f'(x) = \exp(x) 
```


```math
f(x)=\ln(x), \;\; f'(x) = \frac{1}{x}
```

```math
f(x)=\sin(x), \;\; f'(x) = \cos(x) 
```
"""

# ╔═╡ 5568a672-a68d-42ce-ad7d-e2f93f597a19
md"""

## Calculate derivative -- chain rule

Composite two functions ``f_2, f_1`` together, the composite function denoted as ``f_2 \circ f_1``

```math
(f_2 \circ f_1) (x) \triangleq f_2(f_1(x))
```


The derivative is

```math
\large
\frac{d f_1}{dx} = \frac{d f_1}{d f_2} \frac{d f_2}{d x}
```


##

**Example:**


```math
\large
f(x) = (b- ax)^2
```

* as a dependence gragh

```math
\Large
x \textcolor{blue}{\xrightarrow{b-ax}} f_1(x) \textcolor{red}{\xrightarrow{x^2}}f_2(f_1(x))
```

```math
\Large
x \textcolor{blue}{\xrightarrow{b-ax}} (b-ax) \textcolor{red}{\xrightarrow{x^2}} (b-ax)^2 
```



"""

# ╔═╡ 20735597-3cb2-4514-b015-cf12464fb286
md"""


##

**Example:**


```math
\large
f(x) = (b- ax)^2
```

* as a dependence gragh

```math
\Large
x \textcolor{blue}{\xrightarrow{b-ax}} f_1(x) \textcolor{red}{\xrightarrow{x^2}}f_2(f_1(x))
```

```math
\Large
x \textcolor{blue}{\xrightarrow{b-ax}} (b-ax) \textcolor{red}{\xrightarrow{x^2}} (b-ax)^2 
```



* chain rule tells us to _multiply all local derivatives_ 

```math
\Large
x \textcolor{blue}{\xleftarrow{\frac{{d} f_1}{{d} x}}} f_1(x) \textcolor{red}{\xleftarrow{\frac{{d} f_2}{{d} f_1}}}f_2(f_1(x))
```

```math
\Large
x \textcolor{blue}{\xleftarrow{-a}} f_2(x) \textcolor{red}{\xleftarrow{2(b-ax)}}f_2(f_1(x))
```

* the derivative is the multiplication of the local derivatives

```math
\large
\frac{d f}{d x} = \textcolor{red}{\underbrace{2(b-ax)}_{df_1/df_2}} \cdot \textcolor{blue}{\underbrace{(-a)}_{df_2/dx}}
```
"""

# ╔═╡ 9367ff5b-cc1d-41b9-8a06-fe248b0b8a19
md"""

## Differentiation and linear approximation


> If ``f: \mathbb R \rightarrow \mathbb R`` is differentiable at ``x_0``, then
> 
> ``f(x)`` can be locally approximated by a linear function
> ```math
> \Large
> \begin{align}
> f(x) &\approx f(x_0) + f'(x_0)(x-x_0) 
> \end{align}
> ```
"""

# ╔═╡ 6161543b-34d1-44df-9b04-e2644adb3882
Foldable("More formally", md"""


> ```math
> f(x) = f(x_0) + f'(x_0)(x-x_0)  + o(|x-x_0|)
> ```

where the small ``o`` denotes that the function is an order of magnitude smaller around 𝑥0 than the function ``|x -x_0|``.

""")

# ╔═╡ ec39f15f-14af-48c6-beae-9a128c3eccb7
f(x) = x * sin(x^2) + 1; # you can change this function!

# ╔═╡ 1f102762-3f4a-4895-a26d-d44e2804f6de
@bind x̂ Slider(-2:0.2:3, default=-1.5, show_value=true)

# ╔═╡ f59d2e5d-4f1b-4c45-b312-bcfc72f97c75
plt_linear_approx = begin
    # Plot function
    xs = range(-2, 3, 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label=L"$f(x)$",
        xlabel=L"x",
        legend=:topleft,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=10,
		lw = 2,
		ratio = .7,
		framestyle=:zerolines
    )

    # Obtain the function 𝒟fₓ̃ᵀ
    ŷ, 𝒟fₓ̂ᵀ = Zygote.pullback(f, x̂)

    # Plot Dfₓ̃(x)
    # plot!(p, xs, w -> 𝒟fₓ̂ᵀ(w)[1]; label=L"Derivative $\mathcal{D}f_\tilde{x}(x)$")
    # Show point of linearization
    vline!(p, [x̂]; style=:dash, c=:gray, label=L"x_0")
    # Plot 1st order Taylor series approximation
    taylor_approx(x) = f(x̂) + 𝒟fₓ̂ᵀ(x - x̂)[1] # f(x) ≈ f(x̃) + 𝒟f(x̃)(x-x̃)
    plot!(p, xs, taylor_approx; label=L"Linear approx. at $x_0$", lc=2,  lw=2)
end

# ╔═╡ d85e3b50-8fe7-4178-a7b5-3c757dce9677
md"""

## Optimisation
"""

# ╔═╡ 350de8a8-fda9-471a-be21-d5606de38f97
TwoColumn(md"""

Whenenver 

```math
\large 
\frac{\mathrm{d}f}{\mathrm{d}x}(x) =0,
``` 

* it implies ``f(x)`` is flat near ``x``
* the derivative vanishes: it does not increase nor decrease
  * it can be a *maximum*, 
  * a *minimum* 
  * or a *saddle point* (not shown here)
""", md"
![](https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/05-example-monotonicity-derivatives.png)
")

# ╔═╡ dc1f2d6a-5195-4f01-9b1f-350566bea0b9
md"[Figure source](https://tivadardanka.com/book)"

# ╔═╡ 27f70663-d4c9-4a06-aa15-db9e9e4d822c
md"""


## Optimisation

To optimise (maximise or minimise) ``f``, *i.e.*

```math
\Large 
x' \leftarrow \arg\max_x f(x)\;\; \text{or}\;\; x' \leftarrow \arg\min_x f(x)
```

We need to solve

```math
\Large
\frac{\mathrm{d}f}{\mathrm{d}x}(x) = 0

```
* either _analytically_
* or iteratively (gradient descent, more on this later in the course)
"""

# ╔═╡ a646aefe-d2a0-4f8b-bb63-4ffe2ec43ff0
md"""

## Example

To optimise 

```math
f(x) = ax^2 + bx +c
```

Find derivative and set to zero:

```math
f'(x) = 2a\cdot x + b =0 \Rightarrow x = \frac{-b}{2a}
```


"""

# ╔═╡ 5b02277a-1c2e-418e-b4a3-9bed86230cd7
md" ``x=``$(@bind x₀_ Slider(-6.5:0.1:4.5, default= -2/2*1))"

# ╔═╡ f56d2626-ff15-4b4c-8184-7147b58ed7db
let
	gr()
	a, b, c = 1, 2, 20
	f(x) = a* x^2 + b*x+c
	df(x) = 2a * x + b
	xs = range(-6, 4.5, 50)
	plt = plot(xs, f, label=L"f(x)= x^2 + 2x +20", legend=:topleft, lw=2, framestyle=:origin, size=(500,400))
	linear_approx_f(x; f, ∇f, x₀) = f(x₀) + ∇f(x₀) * (x- x₀)
	if df(x₀_) < 0
		plot!((x) -> linear_approx_f(x; f=f, ∇f= df, x₀ = x₀_), legend=:topleft, label="Local linear approx",  lc=:red, lw=1.5, ylim=[0, 50])
	elseif df(x₀_) > 0
		plot!((x) -> linear_approx_f(x; f=f, ∇f= df, x₀ = x₀_), legend=:topleft, label="Local linear approx", lc=:green,  lw=1.5, ylim=[0, 50])
	else
		plot!((x) -> linear_approx_f(x; f=f, ∇f= df, x₀ = x₀_), legend=:topleft, label="Local linear approx", lc=:gray,  lw=1.5, ylim=[0, 50])
	end
	fprime = 2*a*x₀_ + b
	annotate!(x₀_, f(x₀_)-3, L"x_0 = %$(round(x₀_, digits=2));\;\;\; f'(x_0)=%$(round(fprime, digits=2))")
	
	# x₀ = -b/2a
	# # scatter!([x₀], [f(x₀)])
	# scatter!([x₀], [0], label="")
	# plot!([x₀], [f(x₀)], st=:sticks, line=:dash, c=:gray, lw=2, label="")
	# old_xticks = xticks(plt)[1]
	# new_xticks = ([x₀], ["\$-\\frac{b}{2a}=-1\$"])
	# keep_indices = findall(x -> all(x .≠ new_xticks[1]), old_xticks[1])
	# merged_xticks = (old_xticks[1][keep_indices] ∪ new_xticks[1], old_xticks[2][keep_indices] ∪ new_xticks[2])
	# xticks!(merged_xticks)
end

# ╔═╡ 9b94af8d-b5d2-4c57-84ca-9406fa9e2d7b
md"""
## Recap: sample mean as _Projection_

The sample mean of ``\mathbf{d} = \{d_1, d_2\ldots, d_n\}`` is


```math
\large
\bar{d} = \frac{1}{n} \sum_i d_i
```
* it *compresses* a bunch of number into one scalar

"""

# ╔═╡ 404d8f96-c76d-48e8-ae2f-28160fc5c549
begin
	Random.seed!(2345)
	sample_data = sort(randn(8) * 2)
	μ̄ = mean(sample_data)
end;

# ╔═╡ 47b5c1dd-52c9-4670-8203-32cf0c3a0bfb
let
	gr()
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

# ╔═╡ 62efcc68-e897-480b-9f54-1cec0255d35b
md"""
## Sample mean as _Projection_

> **_Sample mean_** is actually a **_projection_**
> * data vector ``\mathbf{d}`` projected to the one vector ``\mathbf{1}``

Because


```math
\large
\frac{\mathbf{1}^\top \mathbf{d}}{\mathbf{1}^\top\mathbf{1}} = \frac{\sum_i d_i}{n} = \bar{{d}}
```


$(aside(tip(md"Recall the definition of projection:

> ```math
> \large
> \mathbf{b}_{\text{proj}}  = \frac{\mathbf{a}^\top\mathbf{b}}
> {\mathbf{a}^\top\mathbf{a}} \mathbf{a}
>```
> * it projects ``\mathbf{b}`` to ``\mathbf{a}``


")))

Multiply ``\mathbf{1}`` on both side, we have

```math
\large 
\frac{\mathbf{1}^\top \mathbf{d}}{\mathbf{1}^\top\mathbf{1}}\mathbf{1} =\bar{{d}}\mathbf{1} =\begin{bmatrix} \bar{d} \\\bar{d} \\ \vdots\\\bar{d}\end{bmatrix}
```

* ``\mathbf{d}``'s **projection** on ``\mathbf{1}``! 


"""

# ╔═╡ 14503fd1-8a66-45d1-b72c-155995aa885e
md"""

## Sample mean as optimisation


We can solve the problem by using **calculus** as well, *i.e.* optimisation


Consider the sum of squared error loss function:

```math
\large
\ell(\mu) = \sum_{i=1}^n (d_i - \mu)^2

```
"""

# ╔═╡ d760db9b-5c5e-4e0b-8e37-a680240f351d
md"``\mu``= $(@bind μ Slider(range(extrema(sample_data)..., 50), default=μ̄))"

# ╔═╡ b324f27a-ccce-4ba4-8c04-6bd03ab11267
let
	gr()
	ylocations = 0.1 * ones(length(sample_data))
	μ̄ = μ
	plt = plot(ylim = [0., 0.15], xminorticks =5, yticks=false, showaxis=:x, size=(650,200), framestyle=:origin)
	δ = 0.1
	for i in 1:length(sample_data)
		plot!([sample_data[i]], [ylocations[i]], label="", markershape =:circle, markersize=5, markerstrokewidth=1, st=:sticks, c=1, annotations = (sample_data[i], ylocations[i] + 0.01, Plots.text(L"d_{%$i}", :bottom, 13)))
		# annotate!([sample_data[i]].+7*(-1)^i * δ, [ylocations[i]].+ δ, "", 8)
	end
	# vline!([μ̄], lw=2, ls=:dash, label="sample mean", legend=:topleft)
	plot!([μ̄], [ylocations[1]], label="", markershape =:star5, markersize=5, markerstrokewidth=1, st=:sticks, c=2, annotations = (μ̄, ylocations[1] + 0.01, Plots.text(L"\mu", :bottom, 15)))
	# density!(scientist_data, label="")

	for idx = 1:8
		plot!([μ̄, sample_data[idx]], 0.1 * idx * [ylocations[1], ylocations[1]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1, 1),  st=:path, label="")
		if isodd(idx)
			annotate!(.5 * [μ̄ + sample_data[idx]], 0.1 * idx *[ylocations[1]], text(L"d_%$(idx) -\mu", 10, :bottom))
		end

		if iseven(idx)
			annotate!(.5 * [μ̄ + sample_data[idx]], 0.1 * idx *[ylocations[1]], text(L"d_%$(idx) -\mu", 10, :bottom))
		end
		# idx = 7
		# plot!([μ̄, sample_data[idx]], .5*[ylocations[1], ylocations[1]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1, 1), st=:path, label="")
	
		# annotate!(.5 * [μ̄ + sample_data[idx]], .5*[ylocations[1]], text(L"d_%$(idx) -\mu", :bottom))
	end


	loss = sum((μ .- sample_data).^2)

	plot!(title=L"\ell = %$(round(loss; digits=2))")
	plt
end

# ╔═╡ 10212f8f-158e-40c8-9b84-5319de464e90
md"""

## Alternative: projection view

"""

# ╔═╡ 6f6fa322-2724-4cf6-8301-229308623cfd
data = [-1, 2.0];

# ╔═╡ e04e96d3-85d6-47b0-be5c-37d3b34c1fec
md"``\mu``= $(@bind μ_ Slider(-2:0.1:3, default=mean(data)))"

# ╔═╡ e1bf8a58-4046-41e8-a0a0-dd45cbd81d75
proj(x::Vector{T}, a::Vector{T}) where T <: Real = dot(a,x)/dot(a,a) * a ; # project vector x to vector a in Julia

# ╔═╡ 02386433-a1d1-4001-8d6c-e16c00ce114a
md"""

**Projection** implies ``\Rightarrow`` **shortest length** of the error vector

```math
\Large
\vec{\text{error} }= \mathbf{d} - \mu \mathbf{1} = \begin{bmatrix}d_1  \\ d_2  \\\vdots \\d_n \end{bmatrix} - \mu \begin{bmatrix}1  \\ 1  \\\vdots \\1 \end{bmatrix} = \begin{bmatrix}d_1-\mu  \\ d_2 -\mu \\\vdots \\d_n -\mu\end{bmatrix}
```

and its length is


```math
\Large
||\mathbf{d} - \mu\mathbf{1}||^2= (\mathbf{d} - \mu\mathbf{1})^\top (\mathbf{d} - \mu\mathbf{1})
```

"""

# ╔═╡ de98855b-b75c-4c72-ae8e-c2716ad08998
md"""
## 

Minimising the length leads to the same loss

```math
\Large
\begin{align}
\arg\min_{\mu} \boxed{||\mathbf{d} - \mu\mathbf{1}||^2  } =\arg\min_\mu \boxed{\sum_i (d_i -\mu)^2}
\end{align}
```

"""

# ╔═╡ 4ccfc448-7fd9-4e79-aa99-51c1163fd09d
md"""

## Sample mean as optimisation


```math
\large
\begin{align}
\hat{\mu} &= \arg\min_{\mu} \ell(\mu) \\
&=\arg\min_{\mu}\sum_{i=1}^n (\mu -d_i)^2
\end{align}
```


Take the derivative and set to zero!


```math
\Large
\ell'(\mu) = 2\sum_{i=1}^n(\mu -d_i)  =0
```
"""

# ╔═╡ 12f636b9-aa89-4a1a-83ba-7c3219289e1f
md"""

## Sample mean as optimisation


```math
\large
\begin{align}
\hat{\mu} &= \arg\min_{\mu} \ell(\mu) \\
&=\arg\min_{\mu}\sum_{i=1}^n (\mu -d_i)^2
\end{align}
```


Take the derivative and set to zero!


```math
\Large
\begin{align}
\ell'(\mu) &= 2\sum_{i=1}^n(\mu -d_i)  =0 \\

\Rightarrow & \sum_{i=1}^n \mu = \sum_{i=1}^nd_i \\
\Rightarrow & \mu = \frac{1}{n} \sum_{i=1}^n d_i
\end{align}
```
"""

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

# ╔═╡ 048a1d00-2268-40e3-b01f-ee23b78eed90
let
	gr()
 	plot( ratio=1, framestyle=:origin)
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	a = [1,1]
	b = data
	# bp = dot(a,b)/dot(a,a)*a
	bp = μ_ * a 
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lc=2, lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lc=1, lw=2)
	plot!([b[1], bp[1]], [b[2],bp[2]], ls=:solid, lc=:gray, lw=2, arrow=true, label="")

	quiver!([0], [0],  quiver=([bp[1]], [bp[2]]), ls=:dash, lw=2)
	annotate!(a[1],a[2], text(L"\mathbf{1}", 15, :top, :red))
	annotate!(b[1],b[2], text(L"\mathbf{d}", 15, :bottom, :blue))
	# annotate!(bp[1]+0.2,bp[2], text(L"b_{\texttt{proj}} =latexify(:(x = $t))", :left))
	if μ_ ≈ mean(data)
		plot!(perp_square(bp, a, b-bp; δ=0.1), lw=1, label="", fillcolor=false)
	end
	annotate!(bp[1]+0.2,bp[2], text(L"\hat{\mathbf{d}} = \mu\mathbf{1}", 15,:left, :purple))
	annotate!(.5 *(data[1] + bp[1]), .5 *(data[2] + bp[2]), text(L"\mathbf{d} - \mu\mathbf{1}", 15, :right, :gray))

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

julia_version = "1.9.0"
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
version = "5.7.0+0"

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
# ╟─7091d2cf-9237-45b2-b609-f442cd1cdba5
# ╟─0a7f37e1-51bc-427d-a947-31a6be5b765e
# ╟─a696c014-2070-4041-ada3-da79f50c9140
# ╟─595a5ef3-4f54-4502-a943-ace4146efa31
# ╟─bc1ee08d-9376-44d7-968c-5e114b09a5e0
# ╟─992a13dd-e6bf-4b18-8654-ac70398e15ab
# ╟─49815f5b-f5e4-4cd4-b891-599033fe9d8b
# ╟─9a2d12cc-59d7-42ef-b1bd-bc90f7c0db3c
# ╟─15428895-7f94-41e4-9fe8-ae2231900afc
# ╟─3de289ab-a863-43b3-9799-3ca66791e02c
# ╟─20536535-dd93-4987-886c-5d1d3cccf469
# ╟─c1a2cdac-605c-4891-abe3-a7e013f390cc
# ╟─d41c8ca2-a6fa-495e-b8c0-5fe6007f2485
# ╟─3ed1700f-4335-4e2d-b0c6-8a121784e38a
# ╟─b4390e22-bf19-4445-8988-954615ac5991
# ╟─1dd3214e-de41-4f9d-b43a-35b107c64cf2
# ╟─d2e4fb88-3728-4886-ba4f-634050bbf738
# ╟─cf1b8de7-1c04-46a8-8ca6-8c266bc7a6fc
# ╟─da2f5399-32de-4998-83f2-b84b2d720f82
# ╟─16ecc090-613e-4746-b12d-0a3d0e4e1727
# ╟─4ef8df63-5b77-48a8-99c0-f014cf6360c1
# ╟─e4bd5842-e6af-4e12-af4b-1556b91db0ee
# ╟─55b1388f-c368-4d95-9893-ea95e8c2359e
# ╟─6fc139a6-a8b6-4215-a4c6-06a54c2985ec
# ╟─a79f418f-d054-46e7-bb9b-f13a7631e21b
# ╟─8ae96192-2b4f-48e7-b55d-bea27d565671
# ╟─bbed688d-0caf-4f93-a19c-ea3a0039a1a2
# ╟─5568a672-a68d-42ce-ad7d-e2f93f597a19
# ╟─20735597-3cb2-4514-b015-cf12464fb286
# ╟─9367ff5b-cc1d-41b9-8a06-fe248b0b8a19
# ╟─6161543b-34d1-44df-9b04-e2644adb3882
# ╠═ec39f15f-14af-48c6-beae-9a128c3eccb7
# ╟─1f102762-3f4a-4895-a26d-d44e2804f6de
# ╟─f59d2e5d-4f1b-4c45-b312-bcfc72f97c75
# ╟─d85e3b50-8fe7-4178-a7b5-3c757dce9677
# ╟─350de8a8-fda9-471a-be21-d5606de38f97
# ╟─dc1f2d6a-5195-4f01-9b1f-350566bea0b9
# ╟─27f70663-d4c9-4a06-aa15-db9e9e4d822c
# ╟─a646aefe-d2a0-4f8b-bb63-4ffe2ec43ff0
# ╟─5b02277a-1c2e-418e-b4a3-9bed86230cd7
# ╟─f56d2626-ff15-4b4c-8184-7147b58ed7db
# ╟─9b94af8d-b5d2-4c57-84ca-9406fa9e2d7b
# ╟─404d8f96-c76d-48e8-ae2f-28160fc5c549
# ╟─47b5c1dd-52c9-4670-8203-32cf0c3a0bfb
# ╟─62efcc68-e897-480b-9f54-1cec0255d35b
# ╟─14503fd1-8a66-45d1-b72c-155995aa885e
# ╟─d760db9b-5c5e-4e0b-8e37-a680240f351d
# ╟─b324f27a-ccce-4ba4-8c04-6bd03ab11267
# ╟─10212f8f-158e-40c8-9b84-5319de464e90
# ╟─6f6fa322-2724-4cf6-8301-229308623cfd
# ╟─e04e96d3-85d6-47b0-be5c-37d3b34c1fec
# ╟─048a1d00-2268-40e3-b01f-ee23b78eed90
# ╟─e1bf8a58-4046-41e8-a0a0-dd45cbd81d75
# ╟─02386433-a1d1-4001-8d6c-e16c00ce114a
# ╟─de98855b-b75c-4c72-ae8e-c2716ad08998
# ╟─4ccfc448-7fd9-4e79-aa99-51c1163fd09d
# ╟─12f636b9-aa89-4a1a-83ba-7c3219289e1f
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
