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
	using StatsPlots
	using LogExpFunctions
end

# ╔═╡ 50752620-a604-442c-bf92-992963b1dd7a
using Images

# ╔═╡ 3b4a2f77-587b-41fd-af92-17e9411929c8
using GaussianProcesses

# ╔═╡ 1afdb42f-6bce-4fb1-860a-820d98df0f9d
using Distributions

# ╔═╡ ef112987-74b4-41fc-842f-ebf1c901b59b
using StatsBase

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ 7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
ChooseDisplayMode()

# ╔═╡ bc96a33d-9011-41ec-a19e-d472cbaafb70
md"""

# CS5914 Machine Learning Algorithms


#### Probability theory
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

# ╔═╡ b67afb1b-8f6b-42e1-8f3a-da8a94f1c864
md"# Motivation"

# ╔═╡ 84916718-b19c-473b-9c39-e1055ccad467
md"""

## Why probability ?

!!! important ""
	##### ``\;\;\;\;\;\;\;\;\;\;`` _Probability theory_ is the tool to deal with **_Uncertainty_** 



"""

# ╔═╡ 1e927760-4dc6-4ecf-86ec-8fed16dbb0d6
md"""

## Why probability ?

!!! important ""
	##### ``\;\;\;\;\;\;\;\;\;\;`` _Probability theory_ is the tool to deal with **_Uncertainty_** 

\
\


!!! note ""
	##### ``\;\;\;\;\;\;\;\;\;\;`` And _**uncertainty**_ is everywhere in Machine Learning



## Uncertainty is everywhere


##### (some) _Sources of uncertainty_: 


* Data is **_Noisy_** 


* _**Model**_ is uncertain


* **_Prediction_** is uncertain


* _**Changing**_ environment


* and more ...
"""

# ╔═╡ 7d8ed44a-06ba-4432-8345-55bb31eb8f1d
md"""

## Prediction is uncertain -- regression
"""

# ╔═╡ 5bd15469-e888-4505-a53d-49fef3329ea4
md"Add linear regression: $(@bind add_lin CheckBox(default=false)),
Add other fits: $(@bind add_gp CheckBox(default=false)),
Add interval: $(@bind add_intv CheckBox(default=false))"

# ╔═╡ c9e0eaae-b340-434e-bdc9-dfdbc747221e
let
	Random.seed!(123)
	# Generate random data for Gaussian process
	nobs = 4
	x = [0.5, 1.5, 4.5, 5.6]
	f(x) =  .75 * x + sin(x)
	y = f.(x) + 0.01 * rand(nobs)
	
	# Set-up mean and kernel
	se = SE(0.0, 0.0)
	m = MeanZero()
	
	# Construct and plot GP
	gp = GP(x, y, m, se, -1e5)

	plt = plot(x, y, st=:scatter, label="", markershape=:circle, markersize= 8,  xlabel=L"x", ylabel=L"y")
	xs = 0:0.05:2π
	plot!(xs, x -> f(x), color=:blue, lw=2, label="true function")
	# plot(gp;  xlabel=L"x", ylabel=L"y", title="Gaussian process", legend=false, xlim =[0, 2π])
	
	samples = rand(gp, xs, 10)
	w0, w1 = [ones(4) x] \ y

	if add_lin
		plot!(xs, (x) -> w0 + w1*x, lw=2, lc=:gray, label="")
	end
	if add_gp
		plot!(xs, samples, lw=2, label="", alpha=0.9)

		if add_intv
			plot!(gp; obsv=false, label="estimation mean")
		end
	end

	plt
end

# ╔═╡ 59a38e32-c2f3-465f-928d-c05f8d69f496
md"""

## Prediction is uncertain -- classification


```math

P(y|\mathbf{X}) = \begin{bmatrix}
\cdot \\
\cdot\\
\cdot\\
\end{bmatrix}
\begin{array}{l}
\texttt{cat}\\
\texttt{dog}\\
\texttt{others}\\
\end{array}
```

"""

# ╔═╡ a7a24713-a29a-4f0c-996b-f98305bac09c
md"""

## Prediction is uncertain -- classification


```math

P(y|\mathbf{X}) = \begin{bmatrix}
\cdot \\
\cdot\\
\cdot\\
\end{bmatrix}
\begin{array}{l}
\texttt{cat}\\
\texttt{dog}\\
\texttt{others}\\
\end{array}
```

"""

# ╔═╡ 2ce6c56b-733c-42e8-a63b-d774cb6c199c
md"""

##
"""

# ╔═╡ 2e4df75b-0778-4ed4-840a-417da2d65204
md"""

##
"""

# ╔═╡ c5be7eb8-e0b6-48cc-8dbe-788fa6624999
Hs_catdogs = ["Cat", "Dog", "Others"];

# ╔═╡ 81ab9972-07bc-4ce9-9138-3359d4e34025
plt1, plt2, plt3=let
	ps = [.9, .05, .05]
	texts = [Plots.text(L"%$(p)", 10) for (i, p) in enumerate(ps)]
	plt_cat = plot(ps, fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", color =:orange,  texts = texts,size=(200,200))


	ps = [.05, .9, .05]
	texts = [Plots.text(L"%$(p)", 10) for (i, p) in enumerate(ps)]
	plt_dog=plot(ps, fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", color =:orange,  texts = texts,size=(200,200))


	ps = [.25, .25, .5]
	texts = [Plots.text(L"%$(p)", 10) for (i, p) in enumerate(ps)]
	plt_dontknow=plot(ps, fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", color =:orange,  texts = texts,size=(200,200))

	plt_cat, plt_dog, plt_dontknow
end;

# ╔═╡ e8fd61f1-33a6-43d8-8056-fb7cf97291b5
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```
$(plt1)

""", md"""

""",
	md"""

""")

# ╔═╡ fc9e9bb6-2287-46c8-8518-c9d0804c094e
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt1)
""", md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/dog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt2)
""",
	md"""


""")

# ╔═╡ 8730b9a2-a1b4-456c-974c-ecd8880e6834
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt1)
""", md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/dog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt2)
""",
	md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/catdog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{???}
```
$(plt3)

""")

# ╔═╡ dc8a3e36-2021-42dd-bc49-0eb6ab784fac
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt1)
""", md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/dog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt2)
""",
	md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/catdog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{???}
```

$(plot([.0, .0, .0], fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", size=(200,200)))
""")

# ╔═╡ ff61cd9d-a193-44b3-a715-3c372ade7f79
md"# Probability theory"

# ╔═╡ 443fa256-ee34-43c0-8efd-c12560c00492
md"""

## What is probability?


There are two views: **Frequentist** and **Bayesian**

> **Frequentist's** interpretation: 
> *  *relative long-term frequency* of something (*i.e.* an **event**) happens
> ```math
> \large
> \mathbb{P}(E) = \lim_{n\rightarrow \infty} \frac{n(E)}{n}
> ```

* ``n(E)``: the count of event ``E`` happens

##



"""

# ╔═╡ 89df7ccb-53da-4b96-bbb4-fe39109467dd
md"Experiment times ``n``: $(@bind mc Slider(1:100000; show_value=true))"

# ╔═╡ bce5c041-be39-4ed1-8935-c389293400bc
penny_image = load(download("https://www.usacoinbook.com/us-coins/lincoln-memorial-cent.jpg"));

# ╔═╡ db6eb97c-558c-4206-a112-6ab3b0ad04c8
begin
	head = penny_image[:, 1:end÷2]
	tail = penny_image[:, end÷2:end]
end;

# ╔═╡ b742a37d-b49a-467b-bda7-6d39cce33125
TwoColumn(md"Coin tossing, **event** `Head`: a head turns up
```math
\large
\mathbb{P}(\texttt{Head}) = \lim_{n\rightarrow \infty}\frac{n(\texttt{head})}{n}=0.5
```

*Frequentist's* interpretation
* toss the coin ``n \rightarrow \infty`` times, 
* half of them will be head. ", [head, tail])

# ╔═╡ ae5c476b-193e-44e3-b3a6-36c8944d4816
begin
	Random.seed!(3456)
	sampleCoins = rand(["head", "tail"], 100000);
	coin_exp = (sampleCoins .== "head")
	nhead = sum(coin_exp[1:mc])
end;

# ╔═╡ f44b5a95-95b5-4d88-927a-61d91ed51c53
sampleCoins[1:mc];

# ╔═╡ 221b8f09-6727-4613-8b96-02d70d337280
L"P(\texttt{head}) = \lim_{n\rightarrow \infty}\frac{n(\texttt{head})}{n} \approx \frac{%$nhead}{%$mc} = %$(round(nhead/mc; digits=3))"

# ╔═╡ 340c9b5b-5ca0-4313-870e-912d5c2dc451
let
	p_mc = nhead/mc
	plot(["tail", "head"], [1-p_mc, p_mc], st=:bar, label="", title=L"P(\texttt{Head})",ylabel="Probability")
end

# ╔═╡ deedb2db-8483-4026-975f-3d5af5a249b7
md"""
## What is probability ?  -- Bayesian

> **Bayesian's** interpretation
> * **subjective** *belief* on something uncertain 

For our coin tossing example,

```math
	\mathbb{P}(\texttt{Head}) = 0.5
``` 

* subjectively, *no preference* in terms of coin's toss will be _head_ or _tail_

##


Both **interpretations** are valid and useful 
* for many cases, *e.g.* time series, the Bayesian interpretation is more natural
  * what is the chance of Brexit?
  * how can we travel back in time to 2018 again and again, infinite times?
"""

# ╔═╡ 128b6ad8-aa21-4d0a-8124-2cf433bc79c4
md"""

## Random variable 


Let's consider ``\cancel{\textbf{random}} \textbf{variable}``  first, 

> e.g. a variable ``\large X= 5``

* ``X``: a _deterministic_ variable
* *with ``100\%`` certainty*, ``X`` takes the value of 5


##


``{\textbf{Random}}\; \textbf{variable}``: the value is _**random**_ rather than _**certain**_



*For example*: $X$ is the rolling realisation of a 6--faced die 🎲
* ``X \in \{1,2,3,4,5,6 \}``

* and ``X`` has a *probability distribution* ``P(X)``
```math
\large
\begin{equation}  \begin{array}{c|cccccc} 
 & X = 1 & X = 2 & X = 3 & X = 4 & X = 5 & X = 6 \\
\hline
P(X) & 1/6 & 1/6 & 1/6 & 1/6 & 1/6 & 1/6

\end{array} \end{equation} 

```
"""

# ╔═╡ 403af436-d7f2-43c0-803a-8104ba69fcfd
md"""
## Probability distributions


Random variables' uncertainties is quantified by **probability distributions**

$$\large P(X=x) \geq 0, \forall x\;\; \text{and}\;\; \sum_x{P(X=x)}=1$$
* non-negative and sum to one


**Temperature** $T: P(T)$

```math

\begin{equation}  \begin{array}{c|c} 
T & P(T)\\
\hline
hot & 0.5 \\
cold & 0.5
\end{array} \end{equation} 
```


**Weather** $W: P(W)$

```math

\begin{equation}  \begin{array}{c|c} 
W & P(W)\\
\hline
sun & 0.6 \\
rain & 0.1 \\
fog & 0.2 \\
snow & 0.1
\end{array} \end{equation} 
```
"""

# ╔═╡ 4d99c216-e32f-43a3-a122-ccbb697711fc
md"""
## Domain of a random variable


**Discrete random variables**
* ``R:`` Is it raining?
  * ``\Omega =\{t, f\}``


* ``T:`` Is the temperature hot or cold?
  * ``\Omega =\{hot, cold\}``


**Continuous random variables**
* ``D \in [0, +\infty)`` How long will it take to drive to St Andrews?
  * ``\Omega = [0, \infty)``


* ``T \in (-\infty, +\infty)`` the temperature in Celcius
  * ``\Omega = [-\infty, \infty)``


We denote random variables with **Capital letters**
* its realisations (values in ``\Omega``) in small cases
"""

# ╔═╡ 5b500acf-7029-43ff-9835-a26d8fe05194
md"""
## Notation

- Capital letter $X,Y, \texttt{Pass}, \texttt{Weather}$ are random variables


- Smaller letters $x,y, +x, -y, \texttt{true, false, cloudy, sunny}$ are particular values r.v.s can take  


- Notation: $P(x)$  is a shorthand notation for $P(X=x)$


- So ``P(X)`` is assumed to be a distribution, but ``P(x)`` is a number

Therefore, $P(W)$ means a full distribution vector

```math

\begin{equation}  \begin{array}{c|c} 
W & P(W)\\
\hline
sun & 0.6 \\
rain & 0.1 \\
fog & 0.2 \\
snow & 0.1
\end{array} \end{equation} 
```

But ``P(sum)`` is a number

```math
P(sun) = P(W=sum) = 0.6 
```


"""

# ╔═╡ 4bf768de-833f-45bf-9429-4820ff61553f
md"""

## Examples of r.v.s

| Variable kind | Discrete or continous| Target space ``\mathcal A`` |
| :---|:---|:---|
| Toss of a coin | Discrete | ``\{0,1\}`` |
| Roll of a die | Discrete |``\{1,2,\ldots, 6\}`` |
| Outcome of a court case | Discrete |``\{0,1\}`` |
| Number of heads of 100 coin tosses| Discrete|``\{0,1, \ldots, 100\}`` |
| Number of covid cases | Discrete|``\{0,1,\ldots\}`` |
| Height of a human | Continuous |``\mathbb{R}^+=(0, +\infty)`` |
| The probability of coin's bias ``\theta``| Continuous|``[0,1]`` |
| Measurement error of people's height| Continuous|``(-\infty, \infty)`` |
"""

# ╔═╡ 656da51f-fd35-4e89-9af5-b5f0fdf8618f
md"""
##  Discrete r.v. -- Bernoulli 




"""

# ╔═╡ 80038fee-b922-479d-9687-771e7e258fcf
md"Model parameter ``\theta``: $(@bind θ Slider(0:0.1:1, default=0.5; show_value=true))"

# ╔═╡ 7c03a15f-9ac1-465f-86a4-d2a6087e5970
TwoColumn(md"""
**Bernoulli random variable**: ``X`` taking binary values ``\{1, 0\}`` 

* for example, coin tossing 
* the distribution _probability mass function_ is 

```math
\large
P(X ) =\begin{cases}\theta & x= 1 \\ 1-\theta & x=0 \end{cases}
```

* ``0\leq \theta \leq 1`` is the parameter of the distribution
* ``0 \leq P(x) \leq 1; \text{and}\; \sum_{x=0,1}P(x) = \theta + 1-\theta = 1``


""", 

	begin
		bar(Bernoulli(θ), xticks=[0,1], xlabel=L"X", ylabel=L"P(X)", label="", ylim=[0,1.0], size=(250,300), title="Bernoulli dis.")
	end
)

# ╔═╡ e28e8089-f52b-440a-9861-895f9c378c84
md"""
## Discrete r.v. -- Bernoulli 
Probability distribution in one--line

```math
\large 
\begin{align}
P(X=x) &=\begin{cases}\theta & x= 1 \\ 1-\theta & x=0 \end{cases} \\

&=\boxed{ \theta^{x} (1-\theta)^{1-x}}
\end{align}
```

``\text{for}\; x\in \{0, 1\}``
* ``x=0``: ``P(X=0) = \theta^{0} (1-\theta)^{1-0} = \underbrace{\theta^0}_{1}\cdot (1-\theta)= 1-\theta``
* ``x=1``: ``P(X=1) = \theta^{1} (1-\theta)^{1-1} = \theta\cdot (1-\theta)^0= \theta``

"""

# ╔═╡ 1e52d388-1e8d-4c20-b6e7-bcdd674ea406
md"""
## Discrete r.v. -- Categorical random variable




"""

# ╔═╡ b662605e-30ef-4e93-a71f-696e76e3ab45
TwoColumn(md"""
``X`` takes categorical values

For example
``\large X\in \{a,b,c\ldots, z, \_\}``
- ``\Omega:`` the English alphabet plus empty space "\_"
- the probability distribution of alphabet in an English text is listed below
    - it tells you $\_$, $\texttt{e, i, n, o}$ are more likely to be used than e.g. letter $\texttt{z}$

""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figure21.png" height="450"/></center>
""")

# ╔═╡ c7210d17-bf91-4434-840f-393eeef1ebd4
md"""

## Discrete r.v. -- Binomial


Toss a coin (**Bernoulli**) with bias (``0\leq \theta \leq 1``) independently ``N`` times: 

```math
 \{Y_1, Y_2, \ldots, Y_n\} \in \{0,1\}^n, \text{and}\;P(Y_i) =\begin{cases}\theta &  Y_i=1 \\ 1-\theta & Y_i=0 \end{cases}
```


"""

# ╔═╡ 4d6badcc-c061-4e63-a156-167376f131eb
md"Total trials: ``n`` $(@bind nb Slider(2:1:100, default=10, show_value=true)),
Bias of each trial ``\theta`` $(@bind θ_bi Slider(0:0.05:1, default=0.7, show_value=true))"

# ╔═╡ 556617f4-4e88-45f4-9d91-066c24473c44
md"""

## Discrete r.v. -- Binomial


Toss a coin (**Bernoulli**) with bias (``0\leq \theta \leq 1``) independently ``N`` times: 

```math
 \{Y_1, Y_2, \ldots, Y_n\} \in \{0,1\}^n, \text{and}\;P(Y_i) =\begin{cases}\theta &  Y_i=1 \\ 1-\theta & Y_i=0 \end{cases}
```


The **total number of heads** (or ``1``s) (or the sum)

$$\large X= \sum_{i=1}^n Y_i$$

## Discrete r.v. -- Binomial


Toss a coin (**Bernoulli**) with bias (``0\leq \theta \leq 1``) independently ``N`` times: 

```math
 \{Y_1, Y_2, \ldots, Y_n\} \in \{0,1\}^n, \text{and}\;P(Y_i) =\begin{cases}\theta &  Y_i=1 \\ 1-\theta & Y_i=0 \end{cases}
```


The **total number of heads** (or ``1``s) (or the sum)

$$\large X= \sum_{i=1}^n Y_i$$

is a **Binomial** distribution; its distribution is

$$\large P(X=x) = \text{Binom}(X; n,\theta)= \binom{n}{x} (1-\theta)^{n-x} \theta^{x},$$


* parameters: ``\theta \in[0,1]`` and ``n \in \mathbb{N}^+``
* ``x\in \{0,1,\ldots, n\}``
* ``\binom{n}{x}``: *binomial coefficient*

## Discrete r.v. -- Binomial


A Binomial r.v. with ``n``= $(nb), ``\theta`` = $(θ_bi) is plotted below
* ``P`` tells how likely you are going to see a result of ``X=x``
  * in this example, the most likely result, called *mode* is 7
  * almost impossible to observe $X=0$, *i.e.* all 10 tosses are tail, the probability is $0.3^{10}$

"""

# ╔═╡ c3910dd8-4919-463f-9af0-bc554566c681
let
	binom = Binomial(nb, θ_bi)
	bar(binom, label="", xlabel=L"X", xticks = 0:nb, ylabel=L"P(X)", title="Binomial with: "*L"n=%$(nb),\;\;\theta = %$(θ_bi)", legend=:topleft)

end

# ╔═╡ f664e72d-e762-4dea-ba11-bc8c6b2863f9
md"""

## Discrete r.v. -- Binomial (conti.)


!!! question "Question"
	Toss a coin (with bias ``0\leq \theta \leq 1``)  ``n`` times
	* what is the probability the number of heads is more than the half?

We can use the Binomial distribution:

```math
\large
P(X \geq \lfloor n/2\rfloor + 1) = \sum_{x \geq \lfloor n/2\rfloor + 1}P(x)
```

* recall ``X`` is the total number of heads
"""

# ╔═╡ c134a0b6-8754-48bd-a2ca-932542744407
let
	binom = Binomial(nb, θ_bi)
	bar(binom, label="", xlabel=L"X", xticks = 0:nb, ylabel=L"P(X)", title=L"n=%$(nb),\;\;\theta = %$(θ_bi)", legend=:topleft)

	p_more_heads = exp(logsumexp(logpdf.(binom, floor(nb/2)+1:1:nb)))
	vspan!([floor(nb/2)+0.5, nb+0.5], alpha=0.5, label=L"P(X \geq \lfloor n/2 \rfloor +1)\approx%$(round(p_more_heads; digits=2))")
end

# ╔═╡ cd746d93-135e-4d10-9a44-50b610344fd9
md"""

## Continuous random variable



**Continuous random variable**: the domain ``\Omega`` is continuous, *e.g.* ``[0,1], (-\infty , +\infty)``
  * ``p:`` *probability density function (p.d.f)*


$$\large \forall x, \;\; p(X=x) \geq 0, \;\; \text{and}\;\; \int_{x\in \Omega}{p(X=x)dx}=1$$

"""

# ╔═╡ a4980317-32aa-44b8-97a8-8887c0e65bb4
md"""

## 	Continuous r.v. -- Uniform


"""

# ╔═╡ 197e2d17-fd19-46b1-8f51-0fa2748340e5
TwoColumn(md"""

``X`` is a **uniform distribution** over interval ``[a, b]``

```math
\large
p(x) = \begin{cases}\frac{1}{b-a} & a \leq x \leq b \\ 0 & \text{otherwise}\end{cases}
```

* denoted as ``X \sim \texttt{Uniform}(a,b)`` if
* no preference over the range between ``a`` and ``b``
**Example**: when ``a=0, b=1``, the distribution reduces to ``p(x) = 1`` for ``x\in [0,1]``



""", 
let
	a, b = 0, 1
	plot(Uniform(0,1), fill=true, alpha= 0.5, lw=2, ylim=[0, 2], xlabel=L"X", ylabel=L"p(X)", label="", title=L"p(X) = \texttt{Uniform}(0,1)", size=(300,300))
end
)

# ╔═╡ 2bad0f9a-2b21-4686-aa41-2b430c354454
md"""

## Probability with p.d.f




"""

# ╔═╡ 9594d76f-3274-48fa-b833-f0c26daa229a
TwoColumn(md"""

\
\
\


!!! question "Question"
	What is the probability that ``X \in [0.5, 1.0]``
""", let
	a, b = 0, 1
	plot(Uniform(0,1), fill=true, alpha= 0.5, lw=2, ylim=[0, 2], xlabel=L"X", ylabel=L"p(X)", label="")
	c = 0.5
	plot!(.5:0.1:1.0, (x)-> 1.0, fill=true, alpha=0.5, label="",title=L"p(X) = \texttt{Uniform}(0,1)", size=(300,300))
end)

# ╔═╡ ce1d7bad-179a-48c0-86c7-2de82c55a96d
md"""
##
"""

# ╔═╡ 09f78f45-3790-4218-847f-b9ea1e61176a
TwoColumn(md"""

We calculate **probability** of continuous r.v. by *integration* (instead of summation)



```math
\begin{align}
P(X \in [0.5, 1.0]) &= \int_{0.5}^{1} p(x) \mathrm{d}x \\
&= \int_{0.5}^{1}1 \mathrm{d}x = 1 \cdot 0.5 = 0.5
\end{align}
```
* interpretation: the shaded area is 0.5

""", let
	a, b = 0, 1
	plot(Uniform(0,1), fill=true, alpha= 0.5, lw=2, ylim=[0, 2], xlabel=L"X", ylabel=L"p(X)", label="")
	c = 0.5
	plot!(.5:0.1:1.0, (x)-> 1.0, fill=true, alpha=0.5, label=L"\mathbb{P}(X\in [0.5, 1.0])=0.5", size=(300,300))
end)

# ╔═╡ b89ac105-597e-44ac-9b58-c1c3c5ac59e9
md"""
## Continuous r.v. -- Gaussian



"""

# ╔═╡ 6bcfe759-0010-400e-bb9b-62c089bd5230
TwoColumn(md"
\

Gaussian random variable $X$

$$\large\begin{align} p(X=x) &= \mathcal{N}(x; \mu, \sigma^2) \\
&= \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x-\mu}{\sigma} \right)^2}
\end{align}$$

* ``x \in (-\infty, +\infty)``
* ``\mu``: mean or location
* ``\sigma^2``: variance or scale, controls the spread

", let

	μs = [-3, 0, 3]
	σ²s = [1 , 2 , 5]
	plt_gaussian = Plots.plot(title="Gaussian distributions", xlabel=L"X", ylabel=L"p(X)", size=(300,300))
	for i in 1:3 
		plot!(plt_gaussian, Normal(μs[i], sqrt(σ²s[i])), fill=true, alpha=0.5, label=L"\mathcal{N}(μ=%$(μs[i]), σ^2=%$(σ²s[i]))")
		vline!([μs[i]], color=i, label="", linewidth=2)
	end
	plt_gaussian
end)

# ╔═╡ d3f51b03-384c-428c-b7e4-bdc1508e6a02
md"""

## Dissect Gaussian

```math
\boxed{\colorbox{lightblue}{$\left(\frac{x -\mu}{\sigma}\right)^2$}} \Longrightarrow -\frac{1}{2}{\left(\frac{x -\mu}{\sigma}\right)^2} \Longrightarrow  \large{e^{-\frac{1}{2}{\left(\frac{x -\mu}{\sigma}\right)^2}} } \Longrightarrow {\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2} {\left(\frac{x -\mu}{\sigma}\right)^2}}
```

* ``\colorbox{lightblue}{$(\frac{x-\mu}{\sigma})^2$}``: the `kernel` measures how far away ``x`` is from ``\mu``
  * measured w.r.t measurement unit ``\sigma``
  * how many ``\sigma`` units away 

"""

# ╔═╡ 875a06b7-eb90-451e-a888-3e4f13832053
md"``\mu``: $(@bind μ1_ Slider(-5:.1:5.0, default=0.0, show_value=true)),
``x``: $(@bind x1_ Slider(-5:.1:5.0, default=2.0, show_value=true))"

# ╔═╡ c2497681-0729-451a-ab5f-43937bc9e100
let

	μ = μ1_
	σ = 1.0

	f1(x) = ((x - μ)/σ )^2

	plot(range(μ -5, μ+5, 100), (x) -> f1(x), lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin)

	x_ = x1_
	plot!([x_, x_], [0, f1(x_)], ls=:dot, lc=1, lw=2, label="")
	annotate!([x_], [f1(x_)], text(L"f(x)=%$(round(f1(x_); digits=2))", :blue,:right))
	annotate!([x_], [0], text(L"x_0", :blue,:top))
	annotate!([μ], [0], text(L"\mu", :red, :top))
	vline!([μ], lc=:red, lw=1.5, label=L"\mu")
end

# ╔═╡ 00c8c5a4-c58f-4a62-ba88-ca3f590977d7
md"""

## Dissect Gaussian

```math
\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow \boxed{\colorbox{orange}{$-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2$} }\Longrightarrow  \large{e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2} } \Longrightarrow {\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}
```


* ``\colorbox{orange}{$-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2$} ``: ``p(x)`` is negative correlated with the distance
  * further away ``x`` is from ``\mu``, ``p(x)`` is smaller
"""

# ╔═╡ 06e178a8-bcd1-4646-8e51-1b90a2e09784
md"``\mu``: $(@bind μ2_ Slider(-5:.1:5.0, default=0.0, show_value=true)),
``x``: $(@bind x2_ Slider(-5:.1:5.0, default=2.0, show_value=true))"

# ╔═╡ ab5612b9-9681-4984-b58e-3783c0c0c6e4
let

	μ = μ2_
	σ = 1

	f1(x) = ((x - μ)/σ )^2
	f2(x) = -0.5* f1(x)
	plot(range(μ -5, μ+5, 100), (x) -> f1(x), lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin)
	plot!(f2, lw=2, label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2")


	x_ = x2_
	plot!([x_, x_], [0, f2(x_)], ls=:dot, lc=:orange, lw=2, label="")
	annotate!([x_], [f2(x_)], text(L"f(x)=%$(round(f2(x_); digits=2))",:orange, :bottom))

	annotate!([x_], [0], text(L"x_0", :orange,:top))
	annotate!([μ], [0], text(L"\mu", :red, :top))

	vline!([μ], lc=:red, lw=2, label=L"\mu")
end

# ╔═╡ 6e7ace1b-6c6f-44e4-8377-dd7804f94ee0
md"""

## Dissect Gaussian

```math
\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow -\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow  \large{ \boxed{\colorbox{lightgreen}{$e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2} $}}} \Longrightarrow {\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}
```


* "``\exp``": the exponential function makes sure ``p(x)>0`` for all ``x``

"""

# ╔═╡ cb3f15a1-3d04-447a-a5a2-50c66f356922
md"``\mu``: $(@bind μ3_ Slider(-5:.1:5.0, default=0.0, show_value=true)),
``x``: $(@bind x3_ Slider(-5:.1:5.0, default=2.0, show_value=true))"

# ╔═╡ 43f6f92c-fe29-484f-ad1b-18a674574ef2
let

	μ = μ3_
	σ = 1

	f1(x) = ((x - μ)/σ )^2
	f2(x) = -0.5* f1(x)
	f3(x) = exp(f2(x))
	# plot(f1, lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2")
	plot(range(μ -5, μ+5, 100), (x) -> f2(x), lw=2, lc=2,label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin)
	plot!(f3, lw=2, lc=3, label=L"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= e^{-\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2}", ylim=[-2,1.5])


	x_ = x3_
	plot!([x_, x_], [0, f3(x_)], ls=:dot, lc=:green, lw=2, label="")
	annotate!([x_], [f3(x_)], text(L"f(x)=%$(round(f3(x_); digits=2))",:green, :bottom))

	vline!([μ], lc=:red, lw=2, label=L"\mu")
end

# ╔═╡ 72af797b-5340-482e-be00-2cda375dd734
md"""

## Dissect Gaussian

```math
\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow -\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow  \large{ e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2} } \Longrightarrow \boxed{\colorbox{pink}{${\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}$}}
```


* ``\frac{1}{\sigma \sqrt{2\pi}}``: normalising constant, a contant from ``x``'s perspective
  * it normalise the density such that $$\int p(x)\mathrm{d}x = 1$$

"""

# ╔═╡ 723365e7-1fad-4899-8ac1-fb8674e2b9a7
md"``\mu``: $(@bind μ4_ Slider(-5:.1:5.0, default=0.0, show_value=true))"

# ╔═╡ a862e9d6-c31d-4b21-80c0-e359a5435b6b
let
	μ = μ4_
	σ = 1
	f1(x) = ((x - μ)/σ )^2
	f2(x) = -0.5* f1(x)
	f3(x) = exp(f2(x))

	f4(x) = 1/(σ * sqrt(2π)) *exp(f2(x))
	# plot(f1, lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2")
	plot(range(μ -5, μ+5, 100), (x) -> f2(x), lw=2, lc=2,label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin)
	plot!(f3, lw=2, lc=3, label=L"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= e^{-\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2}", ylim=[-2,1.5])

	plot!(f4, lw=2, lc=4, label=L"\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2}", ylim=[-2,1.5])

end

# ╔═╡ e44e47c4-fce6-4559-ae32-c315508bbf9c
md"""
## Summary of discrete and continuous r.v.


**Discrete random variable**: ``\Omega`` is discrete, *e.g.* ``\{\texttt{true},\texttt{false}\}, \{1,2,3,\ldots, \infty\}``
  * ``P:`` *probability mass function (p.m.f)* satisfies 
    * ``0\leq P(x) \leq 1``
    * ``\sum_{x\in \Omega} P(x) = 1``
  * *e.g.* Bernoulli, Binomial, Poisson and so on

**Continuous random variable**: ``\Omega`` is continuous, *e.g.* ``[0,1],  (-\infty , +\infty)``
  * ``p:`` *probability density function (p.d.f)* satisfies (smaller case ``p``)
    * ``p(x) \ge 0``
    * ``\int_{x\in \Omega} p(x) {d}x=1``
  * *e.g.* Gaussian, Laplace, Beta distributions and so on
"""

# ╔═╡ e557ad8b-9e4f-4209-908f-2251e2e2cde9
md"""
## Joint distribution


A **joint distribution** over a set of random variables: ``X_1, X_2, \ldots, X_n`` 


```math
\large
\begin{equation} P(X_1= x_1, X_2=x_2,\ldots, X_n= x_n) = P(x_1, x_2, \ldots, x_n) \end{equation} 
```

* the joint event ``\{X_1=x_1, X_2=x_2, \ldots, X_n=x_n\}``'s distribution

* must still statisfy

$P(x_1, x_2, \ldots, x_n) \geq 0\;\; \text{and}\;\;  \sum_{x_1, x_2, \ldots, x_n} P(x_1, x_2, \ldots, x_n) =1$ 

## Joint distribution -- examples


For example, joint distribution of temperature (``T``) and weather (``W``): ``P(T,W)``
```math
\begin{equation}
\begin{array}{c c |c} 
T & W & P\\
\hline
hot & sun & 0.4 \\
hot & rain & 0.1 \\
cold & sun  & 0.2\\
cold & rain & 0.3\end{array} \end{equation} 
```



"""

# ╔═╡ 25c1c999-c9b0-437a-99f3-6bb59482ca7d
md"""

## Joint distribution --  examples

Bi-letter example: $X,Y$ represents the _first_ and _second_ letter
  * *e.g.* $X = \texttt{s}, Y = \texttt{t}$, means a bi--letter "$\texttt{st}$"
"""

# ╔═╡ 1877ebc8-790f-483a-acf3-9288df9ee7cc
TwoColumn(md"""

* there are $27 \times 27$ entries
* very common bigrams are $\texttt{in, re, he, th}$, \_$\texttt{a}$ (starting with ``\texttt{a}``)
* uncommon bigrams are $\texttt{aa, az, tb, j}$\_ (ending with ``\texttt{j}``)
* sum all $27\times 27$ entries will be 1

""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figure19.png" width = "300" height="300"/></center>
""")

# ╔═╡ 66a08217-481a-4956-ba8a-c4c822f3d0d2
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/";

# ╔═╡ 271009dc-c42b-40bc-9896-e6fc16238a73
md"""

## Conditional distribution

Conditional probability is defined as

$$P(A=a|B=b) = \frac{P(A=a, B=b)}{P(B=b)}$$

* read: *probability of ``A`` given ``B``*
* the probability of $A=a$ given $B=b$ is true



"""

# ╔═╡ a01ebe11-dba2-45df-9fe3-1343576c2071
Resource(figure_url * "condiprob.png", :width=>800, :align=>"left")

# ╔═╡ 738c0c4c-d66e-42a5-8b7f-0cb4bb576d18
md"""

## Independence

Random variable $X,Y$ are **independent**, if 

$$\large \forall x,y : P(x,y) = P(x)P(y)$$

*Alternatively*

$$\large \forall x,y: P(x|y) = P(x)$$
  * intuition: knowing (conditional on $Y=y$) does not change the probability of ``X``
  

##

For _multiple_ independent random variables,

```math
P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^n P(X_i)
```
"""

# ╔═╡ 62627b47-5ec9-4d7d-9e94-2148ff198f66
md"""
## Independence: genuine coin toss

> Two sequences of 300 “coin flips” (H for heads, T for tails). 
> 
> * which one is the genuine **independent** coin tosses?

**Sequence 1**

>	TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT

**Sequence 2**

>	HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT

* both of them have ``N_h = 148`` and ``N_t= 152``
"""

# ╔═╡ fc09b97a-13c9-4721-83ca-f7caa5f55079
begin
	seq1="TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT"
	seq2 = "HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT"
	sequence1=map((x) -> x=='H' ? 1 : 2,  [c for c in seq1])
	sequence2=map((x) -> x=='H' ? 1 : 2,  [c for c in seq2])
end;

# ╔═╡ 95779ca4-b743-43f1-af12-6b14c0e28f0b
md"""

## Independence: genuine coin toss (cont.)


Recall **independence**'s definition

```math
\large
P(X_{t+1}|X_{t}) = P(X_{t+1})
```

* ``X_{t}``: the tossing result at ``t``
* ``X_{t+1}``: the next tossing result at ``t+1``


And the conditional distribution should be (due to independence)

```math
\large
P(X_{t+1}=\texttt{h}|X_{t}=\texttt{h}) = P(X_{t+1}=\texttt{h}|X_{t}=\texttt{t}) =P(X_{t+1}=\texttt{h}) = 0.5
```

"""

# ╔═╡ 0f280847-2404-4211-8221-e30418cf4d42
md"""

##


**Sequence 1**

>	TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT

The joint frequency table is

```math

\begin{equation}  \begin{array}{c|cc} 
n(X_{t}, X_{t+1}) & X_{t+1} = \texttt h & X_{t+1} =\texttt t \\
\hline
X_t =\texttt h & 46 & 102 \\ 

X_t= \texttt t & 102 & 49 \\ 

\end{array} \end{equation} 

```

* ``P(X_{t+1}=\texttt h|X_t=\texttt h) =\frac{46}{46+102} \approx 0.311 \ll 0.5``
* ``P(X_{t+1}=\texttt h|X_t=\texttt t) =\frac{102}{102+49} \approx 0.675 \gg 0.5``

"""

# ╔═╡ b682cc8d-4eeb-4ecd-897c-e15a3e40f76d
md"""

##

**Sequence 2**

>	HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT

```math

\begin{equation}  \begin{array}{c|cc} 
n(X_{t}, X_{t+1}) & X_{t+1} = \texttt h & X_{t+1} =\texttt t \\
\hline
X_t =\texttt h & 71 & 77 \\ 

X_t= \texttt t & 76 & 75 \\ 

\end{array} \end{equation} 

```

* ``\hat P(X_{t+1}=\texttt h|X_t=\texttt h) =\frac{71}{71+77} \approx 0.48 \approx 0.5``
* ``\hat P(X_{t+1}=\texttt h|X_t=\texttt t) =\frac{76}{76+75} \approx 0.503 \approx 0.5``
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
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GaussianProcesses = "891a1506-143c-57d2-908e-e1f8e92e6de9"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.95"
GaussianProcesses = "~0.12.5"
HypertextLiteral = "~0.9.4"
Images = "~0.25.3"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.21"
LogExpFunctions = "~0.3.24"
Plots = "~1.38.15"
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
project_hash = "81813cc2e67a3ef9e4385cfc8047332a2fb4a26d"

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
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

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
git-tree-sha1 = "d3f758863a47ceef2248d136657cb9c033603641"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.8"

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

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

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

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

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

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

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
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

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
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

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

[[deps.ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "e1c40d78de68e9a2be565f0202693a158ec9ad85"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.11"

[[deps.ElasticPDMats]]
deps = ["LinearAlgebra", "MacroTools", "PDMats"]
git-tree-sha1 = "5157c93fe9431a041e4cd84265dfce3d53a52323"
uuid = "2904ab23-551e-5aed-883f-487f97af5226"
version = "0.2.2"

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

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

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

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "58d83dd5a78a36205bdfddb82b1bb67682e64487"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.4.9"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "589d3d3bff204bdd80ecc53293896b4f39175723"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.1.1"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "c6e4a1fbe73b31a3dea94b1da449503b8830c306"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.21.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

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

[[deps.GaussianProcesses]]
deps = ["Distances", "Distributions", "ElasticArrays", "ElasticPDMats", "FastGaussQuadrature", "ForwardDiff", "LinearAlgebra", "Optim", "PDMats", "Printf", "ProgressMeter", "Random", "RecipesBase", "ScikitLearnBase", "SpecialFunctions", "StaticArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "31749ff6868caf6dd50902eec652a724071dbed3"
uuid = "891a1506-143c-57d2-908e-e1f8e92e6de9"
version = "0.12.5"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

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

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "b1798a4a6b9aafb530f8f0c4a7b2eb5501e2f2a3"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.16"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SnoopPrecompile", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "d90867cbe037730a73c9a9499b3591eedbe387a0"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "124626988534986113cfd876e3093e4a03890f58"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "e7c68ab3df4a75511ba33fc5d8d9098007b579a8"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.2"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "bcc2f0db769e5d42a67f501ec6781401a4389e8f"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.5"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "44664eea5408828c03e5addb84fa4f916132fc26"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.1"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "ce28c68c900eed3cdbfa418be66ed053e54d4f56"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.7"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "8717482f4a2108c9358e5c3ca903d3a6113badc9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.5"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "5fa9f92e1e2918d9d1243b1131abe623cdf98be7"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.3"

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

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "6667aadd1cdee2c6cd068128b3d226ebc4fb0c67"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.9"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "106b6aa272f294ba47e96bd3acbabdc0407b5c60"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.2"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

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

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

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

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

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

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "5ae7ca23e13855b3aba94550f26146c01d259267"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.0"

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

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "e3a6546c1577bfd701771b477b794a52949e7594"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.6"

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
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "95a4038d1011dfdbde7cecd2ad0ac411e53ab1bc"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.10.1"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

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

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

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

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

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

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

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

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

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
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "14ef622cf28b05e38f8af1de57bc9142b03fbfe3"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.5"

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

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "8621f5c499a8aa4aa970b1ae381aae0ef1576966"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.4"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

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
# ╟─9f90a18b-114f-4039-9aaf-f52c77205a49
# ╟─50752620-a604-442c-bf92-992963b1dd7a
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─7091d2cf-9237-45b2-b609-f442cd1cdba5
# ╟─b67afb1b-8f6b-42e1-8f3a-da8a94f1c864
# ╟─84916718-b19c-473b-9c39-e1055ccad467
# ╟─1e927760-4dc6-4ecf-86ec-8fed16dbb0d6
# ╟─7d8ed44a-06ba-4432-8345-55bb31eb8f1d
# ╟─5bd15469-e888-4505-a53d-49fef3329ea4
# ╟─3b4a2f77-587b-41fd-af92-17e9411929c8
# ╟─c9e0eaae-b340-434e-bdc9-dfdbc747221e
# ╟─59a38e32-c2f3-465f-928d-c05f8d69f496
# ╟─e8fd61f1-33a6-43d8-8056-fb7cf97291b5
# ╟─81ab9972-07bc-4ce9-9138-3359d4e34025
# ╟─a7a24713-a29a-4f0c-996b-f98305bac09c
# ╟─fc9e9bb6-2287-46c8-8518-c9d0804c094e
# ╟─2ce6c56b-733c-42e8-a63b-d774cb6c199c
# ╟─dc8a3e36-2021-42dd-bc49-0eb6ab784fac
# ╟─2e4df75b-0778-4ed4-840a-417da2d65204
# ╟─8730b9a2-a1b4-456c-974c-ecd8880e6834
# ╟─c5be7eb8-e0b6-48cc-8dbe-788fa6624999
# ╟─ff61cd9d-a193-44b3-a715-3c372ade7f79
# ╟─443fa256-ee34-43c0-8efd-c12560c00492
# ╟─b742a37d-b49a-467b-bda7-6d39cce33125
# ╟─89df7ccb-53da-4b96-bbb4-fe39109467dd
# ╟─f44b5a95-95b5-4d88-927a-61d91ed51c53
# ╟─221b8f09-6727-4613-8b96-02d70d337280
# ╟─340c9b5b-5ca0-4313-870e-912d5c2dc451
# ╟─bce5c041-be39-4ed1-8935-c389293400bc
# ╟─db6eb97c-558c-4206-a112-6ab3b0ad04c8
# ╟─1afdb42f-6bce-4fb1-860a-820d98df0f9d
# ╟─ae5c476b-193e-44e3-b3a6-36c8944d4816
# ╟─deedb2db-8483-4026-975f-3d5af5a249b7
# ╟─128b6ad8-aa21-4d0a-8124-2cf433bc79c4
# ╟─403af436-d7f2-43c0-803a-8104ba69fcfd
# ╟─4d99c216-e32f-43a3-a122-ccbb697711fc
# ╟─5b500acf-7029-43ff-9835-a26d8fe05194
# ╟─4bf768de-833f-45bf-9429-4820ff61553f
# ╟─656da51f-fd35-4e89-9af5-b5f0fdf8618f
# ╟─7c03a15f-9ac1-465f-86a4-d2a6087e5970
# ╟─80038fee-b922-479d-9687-771e7e258fcf
# ╟─e28e8089-f52b-440a-9861-895f9c378c84
# ╟─1e52d388-1e8d-4c20-b6e7-bcdd674ea406
# ╟─b662605e-30ef-4e93-a71f-696e76e3ab45
# ╟─c7210d17-bf91-4434-840f-393eeef1ebd4
# ╟─556617f4-4e88-45f4-9d91-066c24473c44
# ╟─4d6badcc-c061-4e63-a156-167376f131eb
# ╟─c3910dd8-4919-463f-9af0-bc554566c681
# ╟─f664e72d-e762-4dea-ba11-bc8c6b2863f9
# ╟─c134a0b6-8754-48bd-a2ca-932542744407
# ╟─cd746d93-135e-4d10-9a44-50b610344fd9
# ╟─a4980317-32aa-44b8-97a8-8887c0e65bb4
# ╟─197e2d17-fd19-46b1-8f51-0fa2748340e5
# ╟─2bad0f9a-2b21-4686-aa41-2b430c354454
# ╟─9594d76f-3274-48fa-b833-f0c26daa229a
# ╟─ce1d7bad-179a-48c0-86c7-2de82c55a96d
# ╟─09f78f45-3790-4218-847f-b9ea1e61176a
# ╟─b89ac105-597e-44ac-9b58-c1c3c5ac59e9
# ╟─6bcfe759-0010-400e-bb9b-62c089bd5230
# ╟─d3f51b03-384c-428c-b7e4-bdc1508e6a02
# ╟─875a06b7-eb90-451e-a888-3e4f13832053
# ╟─c2497681-0729-451a-ab5f-43937bc9e100
# ╟─00c8c5a4-c58f-4a62-ba88-ca3f590977d7
# ╟─06e178a8-bcd1-4646-8e51-1b90a2e09784
# ╟─ab5612b9-9681-4984-b58e-3783c0c0c6e4
# ╟─6e7ace1b-6c6f-44e4-8377-dd7804f94ee0
# ╟─cb3f15a1-3d04-447a-a5a2-50c66f356922
# ╟─43f6f92c-fe29-484f-ad1b-18a674574ef2
# ╟─72af797b-5340-482e-be00-2cda375dd734
# ╟─723365e7-1fad-4899-8ac1-fb8674e2b9a7
# ╟─a862e9d6-c31d-4b21-80c0-e359a5435b6b
# ╟─e44e47c4-fce6-4559-ae32-c315508bbf9c
# ╟─e557ad8b-9e4f-4209-908f-2251e2e2cde9
# ╟─25c1c999-c9b0-437a-99f3-6bb59482ca7d
# ╟─1877ebc8-790f-483a-acf3-9288df9ee7cc
# ╟─66a08217-481a-4956-ba8a-c4c822f3d0d2
# ╟─271009dc-c42b-40bc-9896-e6fc16238a73
# ╟─a01ebe11-dba2-45df-9fe3-1343576c2071
# ╟─738c0c4c-d66e-42a5-8b7f-0cb4bb576d18
# ╟─62627b47-5ec9-4d7d-9e94-2148ff198f66
# ╟─fc09b97a-13c9-4721-83ca-f7caa5f55079
# ╟─ef112987-74b4-41fc-842f-ebf1c901b59b
# ╟─95779ca4-b743-43f1-af12-6b14c0e28f0b
# ╟─0f280847-2404-4211-8221-e30418cf4d42
# ╟─b682cc8d-4eeb-4ecd-897c-e15a3e40f76d
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002