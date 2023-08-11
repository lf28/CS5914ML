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

# ╔═╡ 94a408c8-64fe-11ed-1c46-fd85dc9f96de
begin
	using PlutoTeachingTools
	using PlutoUI
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using LinearAlgebra
	using StatsPlots
	using LogExpFunctions
	using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	using Distributions
end

# ╔═╡ 8fd91a2b-9569-46df-b0dc-f29841fd2015
TableOfContents()

# ╔═╡ cde253e9-724d-4d2b-b82a-e5919240ddd3
ChooseDisplayMode()

# ╔═╡ d5a707bb-d921-4e13-bad0-8b8d03e1852a
md"""

# CS5914 Machine Learning Algorithms


#### Bayes' theorem
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 6df53306-f00a-4d7e-9e77-0b717f016f06
# md"""

# ## Notations


# Superscript--index with brackets ``.^{(i)}``: ``i \in \{1,2,\ldots, n\}`` index observations/data
# * ``n`` total number of observations
# * *e.g.* ``y^{(i)}`` the i-th observation's label
# * ``\mathbf{x}^{(i)}`` the i-th observation's predictor vector

# Subscript--index: feature index ``j \in \{1,2,,\ldots, m\} ``
# * ``m`` total number of features
# * *e.g.* ``\mathbf{x}^{(i)}_2``: the second element/feature of ``i``--th observation


# Vectors: **Bold--face** smaller case:
# * ``\mathbf{x},\mathbf{y}, \boldsymbol{\beta}``
# * ``\mathbf{x}^\top``: row vector

# Matrices: **Bold--face** capital case: 
# * ``\mathbf{X},\mathbf{A}, \boldsymbol{\Gamma}``  


# Scalars: normal letters
# * ``x,y,\beta,\gamma``

# """

# ╔═╡ 65bdd684-09d1-4fdc-b2bd-551f43626812
# md"""

# ## This lecture


# * 
# Maximum likelihood estimation (MLE) has been introduced in **Uncertainty 2**


# In this note, we quickly review MLE and show

# * how least square estimation of regression ``\Leftrightarrow`` MLE
# * MLE is a *general* method to define *goodness*
# * then we justify regularisation with Maximum a Posteriori (MAP) estimator 

# !!! information "Key message"
# 	Probabilistic models unify ML models
#     * loss function ``\Leftrightarrow`` MLE
#     * regularised estimation ``\Leftrightarrow`` MAP (a poor man's Bayesian estimation)
#     * and full Bayesian inference is the best (next time)
# """

# ╔═╡ add26b55-00d5-4dc3-a9d9-d26ef6d2a043
md"""

# Bayes' theorem
"""

# ╔═╡ af71eea9-19b1-41fa-ae71-3433abeab889
md"""

## Bayes' theorem

"""

# ╔═╡ cfc74d6b-8dd6-4d82-b636-e10d39e37e37
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/bayes.png' height = '200' /></center>"

# ╔═╡ a13aa4b1-1e19-4419-94ba-808037b21c7f
md"""

* **Bayes' rule** provides us a way to find out the posterior;
* given the prior and likelihood




## Bayes' theorem -- the evidence term


"""

# ╔═╡ 30938334-8399-4dc5-844e-9af3e27f9b0e
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/bayes4.png' height = '150' /></center>"

# ╔═╡ 4cd836ea-bd0a-435d-8a85-8d999c930316
md"""


And the **Evidence** is the sum of the numerator 

* it serves the purpose of **normalising constant** such that the posterior sum to 1

```math
\large
\sum_h p(h|\mathcal{D}) = 1
```

##

Since the **Evidence** ``p(\mathcal{D})`` is a constant (it does **not** depend on ``h``)
  * we usually skip it and rewrite Bayes' rule with the ``\propto`` 
"""

# ╔═╡ 5244a73c-102e-4620-90af-4c1459c1b7f6
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/bayes2.png' height = '200' /></center>"

# ╔═╡ 9468c098-d0da-4c94-8a71-62e07c6ae6d2
md"""

## Aside: normalisation and ``\propto``

**"Normalisation"**: divide each term by the total sum such that the new sum is one
* example:

$\langle 0.2, 0.3, 0.1 \rangle \stackrel{\text{normalise}}{\Longrightarrow} \left \langle \small \frac{0.2}{0.2+0.3 +0.1}, \frac{0.3}{0.2+0.3+0.1},  \frac{0.1}{0.2+0.3+0.1} \right \rangle = \left \langle \frac{2}{5}, \frac{3}{5}, \frac{1}{5}\right  \rangle$

$\langle 2, 3, 1 \rangle \stackrel{\text{normalise}}{\Longrightarrow} \left\langle \frac{2}{2+3+1}, \frac{3}{2+3+1}, \frac{1}{2+3+1} \right\rangle  = \left\langle \frac{2}{5}, \frac{3}{5},\frac{1}{5} \right\rangle$

``\propto``: **proportional** 

$\langle 0.2, 0.3, 0.1\rangle \propto \langle 2, 3, 1 \rangle$

* the two are equal after normalisation

## ``\propto`` relationship



"""

# ╔═╡ 791f9f85-041b-4dbd-ad8e-a619d8071e20
TwoColumn(md"""
\
\

``\propto``: **proportional** 

* ``\mathbf{x} \propto \mathbf{y}`` means, there is some constant ``\alpha >0`` such that

$$\langle x_1, x_2, \ldots x_n \rangle = \alpha \langle y_1, y_2, \ldots y_n \rangle$$

* *e.g.* $\langle 0.2, 0.3 \rangle  \propto \langle 2, 3 \rangle$, and ``\alpha = ?``

* in other words, they are equal after normalisation

""", let
	f1 = Beta(1.5, 1.5)
	plot(f1,label=L"f_1(x)", lw=2, xlabel=L"x",  size=(350,350))
	plot!((x) -> .5 * pdf(f1, x),label=L"f_2(x)=0.5f_1(x)", lw=2, ls=:dash, xlabel=L"x")
	plot!((x) -> 2 * pdf(f1, x),label=L"f_3(x)=2f_1(x)", lw=2,  ls=:dash,xlabel=L"x", title=L"\propto"*" functions")


	# plot((x) -> log(f1),label=L"f_1(x)", xlabel=L"x")
	# plot!((x) -> .5 * f1(x),label=L"f_2(x)=0.5f_1(x)", xlabel=L"x")
	# plot!((x) -> 2 * f1(x),label=L"f_2(x)=2f_1(x)", xlabel=L"x")
end)

# ╔═╡ 6b9c6600-b35c-4c2e-931d-7c0b94c4c9f3
md"""

## Why Bayes' rule is useful

"""

# ╔═╡ b974be07-b339-459e-a0e9-311265d5029e
TwoColumn(md"""It is always easier for us to think generatively or causally

$$H \xRightarrow[\text{probability}]{\text{forward}} \mathcal{D}$$ 


   * ``h`` is the possible hypothesis (which is usually unknown), with some prior 

$$\text{prior: }P(H)$$
* ``\mathcal{D}`` the observation, or `cough`; given the hidden `h`, how likely you observe the data: 
  
$$P(\mathcal D|H):\;\; \text{likelihood function}.$$

""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/cough_bn.png" width = "100"/></center>
""")

# ╔═╡ 01a34f53-67ee-44e5-ad28-6592892d6041
md"""

## Example: Cough example
##### applying Bayesian theorem

* ``H``: the unknown cause ``h \in \{\texttt{cold}, \texttt{cancer}, \texttt{healthy}\}`` 

* ``\mathcal{D}``: the observed data is ``\mathcal{D} = \texttt{Cough}``
 

Bayes' rule allows us to compute the posterior via ``p(H), p(\mathcal{D}|h)``

```math
P(H|\mathcal{D}) \propto P(H) P(\mathcal{D}|H)
```

* ``\mathcal{D}`` depends on ``H``: the cough observation depends on the hidden cause

* Bayes' rule: compute the reverse probability: ``P(H|\mathcal{D})``

"""

# ╔═╡ 274552b0-c285-4cef-89ab-49795d8b8183
md"""


## Revisit: coughing problem

!!! note ""
	Someone in a café has coughed. Has he got  
	* **cancer** 
	* **cold** 
	* or just **healthy** ?

```math
\large
\text{Hypothesis: }h \in \{\texttt{healthy}, \texttt{cold}, \texttt{cancer}\}
```

```math
\large
\text{Observation: }\mathcal{D} = \{\texttt{cough} = \texttt{true}\}
```


##### But the objective is to find out the posterior

```math
\Large\boxed{P(H|\mathcal{D}) \propto P(H)P(\mathcal{D}|H)}
```

* Bayes' rule comes to rescue

"""

# ╔═╡ bf245f54-d3f4-43e5-895f-f6fbf6d469a4
begin
	Hs = ["healthy", "cold", "cancer"]
	prior_p = [0.89, 0.1, 0.01]
	liks = [0.05, 0.85, 0.9]
	x_= 1
	post_p_unorm = prior_p .* liks
	marg_lik = sum(post_p_unorm)
	post_p = post_p_unorm/marg_lik
end;

# ╔═╡ b3e54667-2745-439d-a442-84a0755bf905
like_plt = plot(liks, fill=true, st=:bar, xticks=(1:3, Hs), markershape=:circ, ylim =[0, 1.0], label="",xlabel=L"h", title="Likelihood: "*L"P(\texttt{cough}=\texttt{t}|h)", c= 2,  lw=2, size=(300,300));

# ╔═╡ 87084518-4e97-4b9a-8356-efbed69ccc5a
md"""## Cough example: prior ``p(H)``


"""

# ╔═╡ aef92673-58df-40d2-8dd0-7a326d707bbc
TwoColumn(md""" 
\
\
\
\

$$\large P(h) = \begin{cases} 0.89 & h=\texttt{healthy} \\
0.1 & h=\texttt{cold}\\ 
0.01 & h=\texttt{cancer}\end{cases}$$ """, 
	begin
prior_plt=plot(prior_p, fill=true, st=:bar, xticks=(1:3, Hs), markershape=:circ, ylim =[0, 1.0], label="", title="Prior: "*L"P(h)", size=(300,300))
end)

# ╔═╡ 8b4ca0a1-307d-456d-bb93-a70f4de1b49f
md"""

## Cough example: likelihood ``p(\mathcal{D}|h)``

"""

# ╔═╡ bec1b8fa-ff3a-4a18-bdd8-4101c671b832
TwoColumn(md"""
\
\
\

The likelihood $$P(\texttt{cough}|h)$$

|  | ``h`` | ``\small P(\texttt{cough}=t\|h)``|``\small P(\texttt{cough}=f\|h)``|
|---|:---:|:---:|:---:|
|  | ``\texttt{healthy}`` | 0.05 | 0.95 |
|  | ``\texttt{cold}`` | 0.85 | 0.15|
|  | ``\texttt{cancer}`` | 0.9 | 0.1|

""", begin
like_plt
end)

# ╔═╡ e08423c0-46b1-45fd-b5ec-c977dfeea575
md"""


## Cough example: posterior

Apply Bayes' theorem to find posterior

$$\begin{align}P(h|\texttt{cough}) = \frac{P(h) P(\texttt{cough}|h)}{P(\texttt{cough})} = \begin{cases} 0.89 \times 0.05/P(\texttt{cough}) & h=\texttt{healthy} \\ 0.1 \times 0.85/P(\texttt{cough}) & h=\texttt{cold} \\ 0.01 \times 0.9/ P(\texttt{cough})& h=\texttt{cancer}  \end{cases}
\end{align}$$



* note that the evidence/marginal likelihood is 

  $$\begin{align}
  P(\texttt{cough}) &= \sum_{h\in \Omega_H} P(h) P(\texttt{cough}|h) \\&= 0.89 \times 0.05 + 0.1 \times 0.85 + 0.01 \times 0.9 = 0.1385\end{align}$$
  is a **normalising constant**: it normalises ``\text{prior} \times \text{likelihood}`` such that the posterior becomes a valid distribution (*i.e.* sum to one)

"""

# ╔═╡ 8a433b8c-e9f3-4924-bdbe-68c8c5d245b5
md"""


## Cough example: posterior

Apply Bayes' theorem to find posterior

$$\begin{align}P(h|\texttt{cough}) = \frac{P(h) P(\texttt{cough}|h)}{P(\texttt{cough})} = \begin{cases} 0.89 \times 0.05/P(\texttt{cough}) & h=\texttt{healthy} \\ 0.1 \times 0.85/P(\texttt{cough}) & h=\texttt{cold} \\ 0.01 \times 0.9/ P(\texttt{cough})& h=\texttt{cancer}  \end{cases}
\end{align}$$



* note that the evidence/marginal likelihood is 

  $$\begin{align}
  P(\texttt{cough}) &= \sum_{h\in \Omega_H} P(h) P(\texttt{cough}|h) \\&= 0.89 \times 0.05 + 0.1 \times 0.85 + 0.01 \times 0.9 = 0.1385\end{align}$$

"""

# ╔═╡ f7875458-fbf8-4ead-aa77-437c11c97550
TwoColumn(md"""
\
\
\
After normalisation

$$P(h|\texttt{cough}) \approx \begin{cases} 0.321 & h=\texttt{healthy} \\
0.614 & h=\texttt{cold}\\ 
0.065 & h=\texttt{cancer}\end{cases}$$ 
""", begin
	# l = @layout [a b; c]
	post_plt = Plots.plot(post_p, seriestype=:bar, markershape=:circle, label="", color=3,  xticks=(1:3, Hs), ylim=[0,1], ylabel="", title="Posterior: "*L"P(h|\texttt{cough})" ,legend=:outerleft, size=(320,300))
	# plt_cough = Plots.plot(prior_plt, like_plt, post_plt, layout=l)
end)

# ╔═╡ d53dc75e-e3ea-4958-80b7-dd32912030ec
md"""

## Summary
"""

# ╔═╡ 3c758419-c188-4158-9452-248af3ac3e83
ThreeColumn(let
	l = @layout [a; b]
	plt_cough = Plots.plot(prior_plt, like_plt, layout=l, size=(280, 400))
end, md"

\
\
\
\
\
\
\

```math
\;\Large \stackrel{\text{update}}{\Longrightarrow}
```", md"""
\
\
$(plot(post_plt, size=(300, 280)))
	""")

# ╔═╡ a57265a3-b55e-4a61-86b9-99c7281c33f8
md"""

## Another example -- change point detection


> Your friend has two coins: 
> * one fair ``p_1= 0.5`` 
> * the other is bent with bias $p_2= 0.2$ 
> He always uses the fair coin at the beginning then switches to the bent coin after some **unknown** number of switches

> We observe the tosses ``\mathcal{D}= [t,h,h,t,h,t,t]``
>
> When did he switch?
"""

# ╔═╡ 17658381-a90e-4f03-8642-1bfa163f8524
md"""
## Thinking generatively

Let's identify the random variables first

* hypothesis: ``S\in \{1,2,\ldots, n-1\}``, the unknown switching point

* data (observation): ``\mathcal{D}=\{d_1, d_2,\ldots\}``

Let's think **forwardly**, or **generatively**

```math
S \Rightarrow {d_1, d_2, \ldots, d_n}
```

* assume knowing ``S\in \{1,2,\ldots, n-1\}`` 

* the likelihood ``P(\mathcal{D}|S)`` is ? 




"""

# ╔═╡ 983fcee3-4e6a-42db-93e4-c535fcbdefa4
md"""
## Forward generative modelling (cont.)


$$P(\mathcal D|S) = \underbrace{\prod_{i=1}^S P(d_i|p_1 =0.5)}_{\text{fair coin}} \underbrace{\prod_{j=S+1}^N P(d_j|p_2 = 0.2)}_{\text{bent coin}};$$ 



  * *e.g.* if ``S=1``

$$P(\{0,1,1,0,1,0,0\}|S=1) = \underbrace{0.5}_{\text{before switch: }\{0\}} \times \underbrace{0.2\cdot 0.2\cdot(1-0.2)\cdot 0.2\cdot (1-0.2)^2}_{\text{after switch: }\{1,1,0,1,0,0\}}$$

  * *e.g.* if ``S=2``

$$P(\{0,1,1,0,1,0,0\}|S=2) = \underbrace{0.5\cdot 0.5}_{\text{before switch: }\{0,1\}} \times \underbrace{0.2\cdot(1-0.2)\cdot 0.2\cdot (1-0.2)^2}_{\text{after switch: }\{1,0,1,0,0\}}$$

* then we can calculate the likelihood for ``S=1,2,\ldots, n-1`` 

"""

# ╔═╡ 074454ba-93fe-4708-952f-0107c4ed43fd
𝒟 = [0,1,1,0,1,0,0];

# ╔═╡ 64899331-8f6e-45af-a52d-1dbad94e75ad
md"""
## Forward generative modelling: Prior for P(S)


To reflect our ignorance, we use a uniform flat prior

$$P(S=s) = \frac{1}{n-1}; \; \text{for }s \in \{1,2,\ldots, n-1\}$$

"""

# ╔═╡ 01aaa24f-cf69-41cb-96cf-c88565f8ec75
begin
	 # to type 𝒟, type "\scrD"+ "tab"
	n1 = length(𝒟)-1
	Plots.plot(1:n1, 1/n1 * ones(n1), st=:sticks, color=1,marker=:circle, label="", ylim=[0,1.0], xlabel=L"S", title=L"P(S)")
	Plots.plot!(1:n1, 1/n1 * ones(n1),  st=:path, fill=true, color=1, alpha= 0.3,  label="Prior")
end

# ╔═╡ cfd53aa2-581d-4d4f-86d3-b901db8722e6
function ℓ_switch(D, p₁=0.5, p₂=0.2)
	likes = zeros(length(D)-1)
	for t in 1:length(likes)
# 		Bernoulli(p) returns an instance of Bernoulli r.v.; coin tosses are Bernoullis!
# 		pdf(Bernoulli(p), y) return the probability, either p or 1-p depends on y
# 		prod times everything together
		likes[t] = prod(pdf.(Bernoulli(p₁), D[1:t])) * prod(pdf.(Bernoulli(p₂), D[(t+1):end]))
	end
	return likes, sum(likes)
end;

# ╔═╡ 9b7c6644-7dbd-4c21-8b30-1b08cbdaad3f
begin
	Plots.plot(1:length(𝒟)-1, ℓ_switch(𝒟)[1], xlabel=L"S", ylabel=L"p(\mathcal{D}|S)", title="Likelihood "*L"p(\mathcal{D}|S)", st=:sticks, marker=:circle, label="")
end

# ╔═╡ dace495d-1a28-495c-908d-1284cd24c244
md"""
## Posterior 
The rest is very routine: **apply Bayes' rule** mechanically

$$P(S|\mathcal D) \propto P(S) P(\mathcal D|S)$$
  * then normalise by dividing by the evidence: ``P(\mathcal D) = \sum_s  P(S)P(\mathcal D|S)``



"""

# ╔═╡ 37425e65-bb8d-4729-bbd1-1bd401a00772
begin
	post_p_switch = ℓ_switch(𝒟)[1]./ℓ_switch(𝒟)[2]
end;

# ╔═╡ a59575a0-0d51-4100-be37-29cc81b9ee3a
begin
	# n1 = length(𝒟)-1
	Plots.plot(1:n1, 1/n1 * ones(n1), st=:sticks, color=1,marker=:circle, label="", legend=:outerbottom, size=(600, 500))
	Plots.plot!(1:n1, 1/n1 * ones(n1),  st=:path, fill=true, color=1, alpha= 0.3,  label="Prior")
	Plots.plot!(1:n1, post_p_switch, xlabel=L"S", ylabel=L"p(S|\mathcal{D})", title="Posterior after update: "*L"P(S|\mathcal{D})", st=:sticks, marker=:circle, color=2, label="")
	Plots.plot!(1:n1, post_p_switch, st=:path, fill=true, alpha=0.3, color=2, label="Posterior")
end

# ╔═╡ 95e497b5-fc4e-4ca6-bb54-abaaaad1fc4d
md"""


## Another dataset ?

"""

# ╔═╡ 9298becb-7b6c-47e0-886e-7ed7835697d3
𝒟₂ = [0,1,1,0,1,0,0, 1,1,1,0,0,0]

# ╔═╡ 6baf37c3-4aa4-4d5d-8c09-addde10a7bc0
begin
	
	plot(findall(𝒟₂ .== 0), 𝒟₂[𝒟₂ .== 0], seriestype=:scatter, label=L"\texttt{tail}", legend=:outerbottom, xlabel="time")
	plot!(findall(𝒟₂ .== 1), 𝒟₂[𝒟₂ .== 1], seriestype=:scatter, label=L"\texttt{head}")
end;

# ╔═╡ 6771b621-b7d0-4a08-b686-554e79a68f56
begin
	post_p₂ = ℓ_switch(𝒟₂)[1]./ℓ_switch(𝒟₂)[2]
end;

# ╔═╡ c1c59f5e-83a3-42d5-b2d2-4139fb24a8c8
let
	post_p = post_p₂
	n1 = length(𝒟₂)-1
	Plots.plot(1:n1, 1/n1 * ones(n1), st=:sticks, color=1,marker=:circle, label="", legend=:outerbottom, size=(600, 500))
	Plots.plot!(1:n1, 1/n1 * ones(n1),  st=:path, fill=true, color=1, alpha= 0.3,  label="Prior")
	Plots.plot!(1:n1, post_p, xlabel=L"S", ylabel=L"p(S|\mathcal{D})", title="Posterior after update: "*L"P(S|\mathcal{D})", st=:sticks, marker=:circle, color=2, label="")
	Plots.plot!(1:n1, post_p, st=:path, fill=true, alpha=0.3, color=2, label="Posterior")
end

# ╔═╡ 32037f42-9fd5-4280-87ee-c708685c49aa
md"""

# MLE and MAP
"""

# ╔═╡ 14b90d1d-d68e-4c15-97ce-2360c06829f8
md"""
## Recap: _Likelihood_ and MLE

!!! information "Likelihood function"
	The likelihood function of ``h`` is defined as 
    ```math
	\Large
		{P(\mathcal{D}|h)}
	```
	* it is a function of ``h``  
      * as ``\mathcal{D}`` is observed therefore fixed
	* **likelihood**: conditional probability of observing the data ``\mathcal D`` given ``h``

"""

# ╔═╡ 87dad92c-0c10-4032-ac1e-5e23a8a4050a
md"""

!!! note "Maximum likelihood estimation (MLE)"

	```math
	\Large
	\hat{h}_{\text{MLE}} \leftarrow \arg\max_{h} P(\mathcal D|h)
	```


"""

# ╔═╡ ec21cda5-f111-470e-97be-66628b71cdb9
md"""
## Recap: MLE overfits
"""

# ╔═╡ fc2c5d6f-d8e1-4a53-9c03-9d52fe82edc7
TwoColumn(md"""
\
\

The likelihood function is

$$P(\{\texttt{cough} = \texttt{t}\}|h) = \begin{cases} 0.05 & h = \texttt{heal.} \\
0.85 & h = \texttt{cold} \\
\colorbox{pink}{0.9} & \colorbox{pink}{h = \texttt{cancer}}
\end{cases}$$



```math
\large
\boxed{
\hat{h}_{\text{MLE}} = \texttt{cancer}}
```

**MLE** suffers **overfitting**!

""",begin
like_plt
end)

# ╔═╡ e172c018-efbe-4ac5-a8ae-c889ea944121
md"""

## Maximum a Posteriori (MAP)



Similar MLE, the **M**aximum **a** **P**osteriori (MAP)



!!! note "Maximum a Posteriori (MAP)"

	```math
	\Large
	\hat{h}_{\text{MAP}} \leftarrow \arg\max_{h} P(h|\mathcal D)
	```


"""

# ╔═╡ e6bbc326-f9b7-4d51-bb15-a2e94b35cb85
TwoColumn(md"""
\
\

The MAP estimator is


$$P(h|\texttt{cough}) \approx \begin{cases} 0.321 & h=\texttt{healthy} \\
\colorbox{pink}{0.614} & \colorbox{pink}{$h=\texttt{cold}$}\\ 
0.065 & h=\texttt{cancer}\end{cases}$$ 

```math
\large
\boxed{
\hat{h}_{\text{MAP}} = \texttt{cold}}
```

**MAP** overcomes **overfitting**!

""",begin
post_plt
end)

# ╔═╡ 3d97fb4f-6c65-44f3-8780-2309516a04f8
md"""

## Maximum a Posteriori (MAP) (with log posterior)



Similar to MLE, we find **M**aximum **a** **P**osteriori (MAP) by taking the ``\ln``



!!! note "Maximum a Posteriori (MAP)"

	```math
	\Large
	\begin{align}
	\hat{h}_{\text{MAP}} &\leftarrow \arg\max_{h}\, \ln P(h|\mathcal D) \\
		&=\, \arg\max_{h}\, \ln \frac{P(h)P(\mathcal D|h)}{p(\mathcal{D})} \\
		&=\, \arg\max_{h}\, \ln P(h) + \ln P(\mathcal D|h) - \underbrace{\ln p(\mathcal{D})}_{\rm constant} \\
		&=\, \arg\max_{h}\, \ln P(h) + \ln P(\mathcal D|h) 
	\end{align}
	```


"""

# ╔═╡ a4bd9881-a68f-4aa7-a7b5-1166d2c371ff
md"


> Note that it avoids computing the ``p(\mathcal{D})``, which can be expensive"

# ╔═╡ 232a99bb-2cc9-41bd-8ea7-38847eaa5ebe
md"""
## Example
"""

# ╔═╡ 372d0121-c6bf-40b8-aa2f-069b8c25b763
md"
Add log-prior: $(@bind add_logprior CheckBox(default=false)), Add log-likelihood: $(@bind add_loglik CheckBox(default=false)), Add log posterior: $(@bind add_logpost CheckBox(default=false))
"

# ╔═╡ bf4b3a28-fbba-44f6-b4f1-751ec5658bfb
TwoColumn(md"""
\
\

The MAP estimator is

$$\begin{align}\arg&\max_h \,\ln P(h|\texttt{cough})\\
&= \arg\max_h\,\{ \colorbox{lightblue}{$\ln p(h)$} +\colorbox{lightsalmon}{$\ln p(\texttt{cough}|h)$}\}
\end{align}$$




```math
\large
\boxed{
\hat{h}_{\text{MAP}} = \texttt{cold}}
```

* it returns the same result

""",begin
	if add_logprior
		plt = plot(log.(prior_p), fill=false, st=:line, xticks=(1:3, Hs), markershape=:x, markerstrokewidth=4, title="Log prior: "*L"\ln P(h)", size=(350,350), alpha=0.5, label="log prior")
	else
		plt = plot(log.(zeros(3)), fill=false, st=:line, xticks=(1:3, Hs), markershape=:x, markerstrokewidth=4, title="Log prior: "*L"\ln P(h)", size=(350,350), alpha=0.5)
	end

	if add_loglik
		plot!(plt, log.(liks), fill=false, st=:line, markershape=:circ,  title="Log-lik: "*L"\ln P(\mathcal{D}|h)", c=2, alpha=0.5, label="log lik")
	else
		plt
	end

	if add_logpost
		plot!(plt, log.(prior_p) + log.(liks), fill=false, st=:line, markershape=:circ, markersize=8, label="log posterior", title="Log-posterior",c=3, alpha=1)
	else 
		plt
	end
end)

# ╔═╡ d57e6f78-3da7-4bbd-a10e-d44b010025a8
md"""

## Summary

"""

# ╔═╡ df7677f8-df25-48d4-8133-ffce71c65a48
md"""

!!! note "Maximum likelihood estimation (MLE)"

	```math
	\Large
	\begin{align}
	\hat{h}_{\text{MLE}} &\leftarrow \arg\max_{h}\, P(\mathcal D|h)\\
		&=\,\arg\max_{h}\, \ln P(\mathcal D|h)
	\end{align}
	```


!!! note "Maximum a Posteriori (MAP)"

	```math
	\Large
	\begin{align}
	\hat{h}_{\text{MAP}} &\leftarrow \arg\max_{h}\, P(h|\mathcal D) \\
		&=\, \arg\max_{h}\, \ln P(h) + \ln P(\mathcal D|h) 
	\end{align}
	```

"""

# ╔═╡ b6af72e6-07d8-4209-aa2e-829ad904a99c
md"""
## Recap: probabilistic linear regression


##### Unknown hypothesis: ``\mathbf{w}``
* assume ``\sigma^2`` is known 

##### The observations: ``\mathcal{D} = \underbrace{\{y^{(1)}, y^{(2)}, \ldots, y^{(n)}\}}_{\mathbf{y}}``

\

##### The likelihood, therefore, is

```math
\large
p(\mathbf{y}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{(i)}\}) = \prod_{i=1}^n p(y^{(i)}|\mathbf{w}, \sigma^2, \mathbf{x}^{(i)})
```

The log likelihood is 

$$\large \begin{align}\ln p(\mathbf{y}|\mathbf{w}, \sigma^2) &= \sum_{i=1}^n \ln p(y^{(i)}|\mathbf{w}, \sigma^2, \mathbf{x}^{(i)})\\
&=  -\frac{n}{2} \ln 2\pi\sigma^2 -\frac{1}{2\sigma^2} \underbrace{\colorbox{pink}{$\sum_{i=1}^n({y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)})^2$}}_{\text{sum of squared error loss!}}
\end{align}$$

"""

# ╔═╡ 1d6fd6a8-1a40-4cb8-8f5b-af9b1f21ea36
md"""
## MAP ⇔ Regularisation



##### Introduce a prior ``p(\mathbf{w})`` 

* assume ``\sigma^2`` known for the moment

* then find the maximum a posteriori (**MAP**) estimator

$$\large p(\mathbf{w}|\mathbf{y}, \mathbf{X}, \sigma^2) \propto \underbrace{p(\mathbf{w})}_{\text{prior}} \underbrace{p(\mathbf{y}|\mathbf{X}, \mathbf{w}, \sigma^2)}_{\text{likelihood}}$$

$$\large \hat{\mathbf{w}}_{\text{MAP}} \leftarrow \arg\max_{\mathbf{w}}\,  \ln  {p(\mathbf{w})} +\ln {p(\mathbf{y}|\mathbf{X}, \mathbf{w}, \sigma^2)}$$

\

!!! note "MAP is cheap"
	Note that we do not need the normalising constant to find the MAP
	* its selling point


## MAP ⇔ Regularisation


Assume ``\mathbf{w}\in \mathbb{R}^m`` is **zero mean ``m``--dimensional independent Gaussians** 

$\large p(\mathbf{w})= \prod_{j=1}^m \mathcal{N}(w_j; 0, 1/\lambda) =  \frac{1}{\left (\sqrt{2\pi\cdot  1/\lambda} \right )^m} \exp{\left \{ - \frac{\lambda}{2} \sum_{j=1}^m w_j^2 \right\}}$

The log prior is


$\large \ln p(\mathbf{w}) =  -\frac{\lambda}{2} \sum_{j=1}^m w_j^2  + C$




"""

# ╔═╡ 41ba6e50-747d-4f75-8d1d-718c7054baff
md"""
## MAP ⇔ Regularisation


Assume ``\mathbf{w}\in \mathbb{R}^m`` is **zero mean ``m``--dimensional independent Gaussians** 

$\large p(\mathbf{w})= \prod_{j=1}^m \mathcal{N}(w_j; 0, 1/\lambda) =  \frac{1}{\left (\sqrt{2\pi\cdot  1/\lambda} \right )^m} \exp{\left \{ - \frac{\lambda}{2} \sum_{j=1}^m w_j^2 \right\}}$

The log prior is


$\large \ln p(\mathbf{w}) =  -\frac{\lambda}{2} \sum_{j=1}^m w_j^2  + C$


Add the log prior and log-likelihood

$$\large
\begin{align}\colorbox{lightblue}{$\ln  {p(\mathbf{w})}$}_{\small\text{log-prior}} &+\colorbox{lightsalmon}{$\ln {p(\mathbf{y}|\mathbf{X}, \mathbf{w}, \sigma^2)}$}_{\small\text{log-lik}} \\
&= \colorbox{lightblue}{$-  \frac{\lambda}{2} \sum_j w_j^2 $} \, \colorbox{lightsalmon}{$-\frac{1}{2}\sum_{i=1}^n (y^{(i)}- \mathbf{w}^\top \mathbf{x}^{(i)})^2$} + C
\end{align}$$




"""

# ╔═╡ 06eb8148-ba1d-4b87-bb3a-57dc44be136f
md"""
## MAP ⇔ Regularisation


Assume ``\mathbf{w}\in \mathbb{R}^m`` is **zero mean ``m``--dimensional independent Gaussians** 

$\large p(\mathbf{w})= \prod_{j=1}^m \mathcal{N}(w_j; 0, 1/\lambda) =  \frac{1}{\left (\sqrt{2\pi\cdot  1/\lambda} \right )^m} \exp{\left \{ - \frac{\lambda}{2} \sum_{j=1}^m w_j^2 \right\}}$

The log prior is


$\large \ln p(\mathbf{w}) =  -\frac{\lambda}{2} \sum_{j=1}^m w_j^2  + C$


Add the log prior and log-likelihood

$$\large
\begin{align}\colorbox{lightblue}{$\ln  {p(\mathbf{w})}$}_{\small\text{log-prior}} &+\colorbox{lightsalmon}{$\ln {p(\mathbf{y}|\mathbf{X}, \mathbf{w}, \sigma^2)}$}_{\small\text{log-lik}} \\
&= \colorbox{lightblue}{$-  \frac{\lambda}{2} \sum_j w_j^2 $} \, \colorbox{lightsalmon}{$-\frac{1}{2}\sum_{i=1}^n (y^{(i)}- \mathbf{w}^\top \mathbf{x}^{(i)})^2$} + C
\end{align}$$




* maximise the above is the same as **minimising** its negative
$$\large
\begin{align}
\colorbox{lightblue}{$ \frac{\lambda}{2} \sum_j w_j^2 $} \, +\colorbox{lightsalmon}{$\frac{1}{2}\sum_{i=1}^n (y^{(i)}- \mathbf{w}^\top \mathbf{x}^{(i)})^2$} + C
\end{align}$$


!!! note ""
	**MAP is just Ridge regression !!!**


"""

# ╔═╡ 908d2e1a-64d4-4bbd-ad25-b8709de83a42
md"""
## MAP ⇔ ``L_1`` Regularisation


Assume ``\mathbf{w}\in \mathbb{R}^m`` is **zero mean ``m``--dimensional independent Laplace** 

$\large p(\mathbf{w})= \prod_{j=1}^m \texttt{Laplace}(w_j; 0, 1/\lambda) =  \frac{1}{\left (2\cdot  1/\lambda \right )^m} \exp{\left \{ - {\lambda} \sum_{j=1}^m |w_j| \right\}}$

The log prior is


$\large \ln p(\mathbf{w}) =  -{\lambda}\sum_{j=1}^m |w_j|  + C$


Add the log-likelihood to the log prior:

$$\large
\begin{align}\colorbox{lightblue}{$\ln  {p(\mathbf{w})}$} &+\colorbox{lightsalmon}{$\ln {p(\mathbf{y}|\mathbf{X}, \mathbf{w}, \sigma^2)}$} \\
&= \colorbox{lightblue}{$-  \lambda \sum_{j=1}^m |w_j|$} \, \colorbox{lightsalmon}{$-\frac{1}{2}\sum_{i=1}^n (y^{(i)}- \mathbf{w}^\top \mathbf{x}^{(i)})^2$} + C
\end{align}$$



!!! note ""
	**MAP is just Lasso regression !!!**


"""

# ╔═╡ 4c7e30b8-5332-4bf3-a23b-4e5c49580ed4
md"""

# Appendix
"""

# ╔═╡ 53a10970-7e9b-4bd4-bd65-b22028b66835
begin
	# define a function that returns a Plots.Shape
	rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
end;

# ╔═╡ 73d51e22-3e40-4cbb-b649-65b719036647
# md"""

# ## (Independent) multivariate Gaussian


# Consider a ``2\times 1`` random vector 

# ```math
# \mathbf{x} = \begin{bmatrix} x_1\\ x_2 \end{bmatrix}
# ```


# If we assume each element is a zero mean univariate Gaussian *i.e.*

# ```math
# x_1 \sim \mathcal{N}(0, \sigma_1^2)\;\;x_2 \sim \mathcal{N}(0, \sigma_2^2)
# ```

# If we further assume they are independent, the joint probability distribution ``p(\mathbf{x})`` is

# $$\begin{align}p(\mathbf{x}) =p(x_1)p(x_2) 
# &= \underbrace{\frac{1}{\sqrt{2\pi}\sigma_1}\exp\left [-\frac{1}{2} \frac{(x_1-0)^2}{\sigma_1^2}\right ]}_{{p(x_1)}} \cdot \underbrace{\frac{1}{\sqrt{2\pi}\sigma_2}\exp\left [-\frac{1}{2} \frac{(x_2-0)^2}{\sigma_2^2}\right ]}_{p(x_2)} \\
# &= \frac{1}{(\sqrt{2\pi})^2 \sigma_1 \sigma_2} \exp{\left \{ -\frac{1}{2} \left (\frac{x_1^2}{\sigma_1^2}+\frac{x_2^2}{\sigma_2^2}\right ) \right\}}
# \end{align}$$


# Generalise the idea to ``n`` dimensional ``\mathbf{x} \in R^n``

# $$\begin{align}p(\mathbf{x}) 
# &= \frac{1}{\left (\sqrt{2\pi} \right )^n \prod_i \sigma_i} \exp{\left \{ -\frac{1}{2} \sum_{i=1}^n\frac{1}{\sigma_i^2} x_i^2 \right\}}
# \end{align}$$
# """

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.98"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.21"
LogExpFunctions = "~0.3.24"
Plots = "~1.38.16"
PlutoTeachingTools = "~0.2.12"
PlutoUI = "~0.7.51"
StatsBase = "~0.34.0"
StatsPlots = "~0.15.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "4cca7d582b35775a4ce38a3267f8f0b90734672c"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "cad4c758c0038eea30394b1b671526921ca85b21"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.4.0"
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

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

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

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "42fe66dbc8f1d09a44aa87f18d26926d06a35f84"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.3"

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
git-tree-sha1 = "4e88377ae7ebeaf29a047aa1ee40826e0b708a5d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.7.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

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
git-tree-sha1 = "cf25ccb972fec4e4817764d01c82386ae94f77b4"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.14"

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
git-tree-sha1 = "e76a3281de2719d7c81ed62c6ea7057380c87b1d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.98"

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

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

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
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "e5556303fd8c9ad4a8fceccd406ef3433ddb4c45"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.4.0"

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
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

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
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "3e008a7aa28d717a5badd05cb70c834e31001077"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.13"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "ce7ea9cc5db29563b1fe20196b6d23ab3b111384"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.18"

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

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "154d7aaa82d24db6d8f7e4ffcfe596f40bff214b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.1.0+0"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

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

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "4b2e829ee66d4218e0cef22c0a64ee37cf258c29"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.1"

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
git-tree-sha1 = "75ca67b2c6512ad2d0c767a7cfc55e75075f8bbc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.16"

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
git-tree-sha1 = "45f9e1b6f62a006a585885f5eb13fc22554a8865"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.12"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

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
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
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
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

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
git-tree-sha1 = "7beb031cf8145577fbccacd94b8a8f4ce78428d3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0da7e6b70d1bb40b1ace3b576da9ea2992f76318"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.0"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "1d5708d926c76a505052d0d24a846d5da08bc3a4"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.1"

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

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

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
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c4d2a349259c8eba66a00a540d550f122a3ab228"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.15.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
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
# ╟─94a408c8-64fe-11ed-1c46-fd85dc9f96de
# ╟─8fd91a2b-9569-46df-b0dc-f29841fd2015
# ╟─cde253e9-724d-4d2b-b82a-e5919240ddd3
# ╟─d5a707bb-d921-4e13-bad0-8b8d03e1852a
# ╟─6df53306-f00a-4d7e-9e77-0b717f016f06
# ╟─65bdd684-09d1-4fdc-b2bd-551f43626812
# ╟─add26b55-00d5-4dc3-a9d9-d26ef6d2a043
# ╟─af71eea9-19b1-41fa-ae71-3433abeab889
# ╟─cfc74d6b-8dd6-4d82-b636-e10d39e37e37
# ╟─a13aa4b1-1e19-4419-94ba-808037b21c7f
# ╟─30938334-8399-4dc5-844e-9af3e27f9b0e
# ╟─4cd836ea-bd0a-435d-8a85-8d999c930316
# ╟─5244a73c-102e-4620-90af-4c1459c1b7f6
# ╟─9468c098-d0da-4c94-8a71-62e07c6ae6d2
# ╟─791f9f85-041b-4dbd-ad8e-a619d8071e20
# ╟─6b9c6600-b35c-4c2e-931d-7c0b94c4c9f3
# ╟─b974be07-b339-459e-a0e9-311265d5029e
# ╟─01a34f53-67ee-44e5-ad28-6592892d6041
# ╟─274552b0-c285-4cef-89ab-49795d8b8183
# ╟─b3e54667-2745-439d-a442-84a0755bf905
# ╟─bf245f54-d3f4-43e5-895f-f6fbf6d469a4
# ╟─87084518-4e97-4b9a-8356-efbed69ccc5a
# ╟─aef92673-58df-40d2-8dd0-7a326d707bbc
# ╟─8b4ca0a1-307d-456d-bb93-a70f4de1b49f
# ╟─bec1b8fa-ff3a-4a18-bdd8-4101c671b832
# ╟─e08423c0-46b1-45fd-b5ec-c977dfeea575
# ╟─8a433b8c-e9f3-4924-bdbe-68c8c5d245b5
# ╟─f7875458-fbf8-4ead-aa77-437c11c97550
# ╟─d53dc75e-e3ea-4958-80b7-dd32912030ec
# ╟─3c758419-c188-4158-9452-248af3ac3e83
# ╟─a57265a3-b55e-4a61-86b9-99c7281c33f8
# ╟─17658381-a90e-4f03-8642-1bfa163f8524
# ╟─983fcee3-4e6a-42db-93e4-c535fcbdefa4
# ╟─074454ba-93fe-4708-952f-0107c4ed43fd
# ╟─9b7c6644-7dbd-4c21-8b30-1b08cbdaad3f
# ╟─64899331-8f6e-45af-a52d-1dbad94e75ad
# ╟─01aaa24f-cf69-41cb-96cf-c88565f8ec75
# ╟─cfd53aa2-581d-4d4f-86d3-b901db8722e6
# ╟─dace495d-1a28-495c-908d-1284cd24c244
# ╟─37425e65-bb8d-4729-bbd1-1bd401a00772
# ╟─a59575a0-0d51-4100-be37-29cc81b9ee3a
# ╟─95e497b5-fc4e-4ca6-bb54-abaaaad1fc4d
# ╠═9298becb-7b6c-47e0-886e-7ed7835697d3
# ╟─6baf37c3-4aa4-4d5d-8c09-addde10a7bc0
# ╟─6771b621-b7d0-4a08-b686-554e79a68f56
# ╟─c1c59f5e-83a3-42d5-b2d2-4139fb24a8c8
# ╟─32037f42-9fd5-4280-87ee-c708685c49aa
# ╟─14b90d1d-d68e-4c15-97ce-2360c06829f8
# ╟─87dad92c-0c10-4032-ac1e-5e23a8a4050a
# ╟─ec21cda5-f111-470e-97be-66628b71cdb9
# ╟─fc2c5d6f-d8e1-4a53-9c03-9d52fe82edc7
# ╟─e172c018-efbe-4ac5-a8ae-c889ea944121
# ╟─e6bbc326-f9b7-4d51-bb15-a2e94b35cb85
# ╟─3d97fb4f-6c65-44f3-8780-2309516a04f8
# ╟─a4bd9881-a68f-4aa7-a7b5-1166d2c371ff
# ╟─232a99bb-2cc9-41bd-8ea7-38847eaa5ebe
# ╟─372d0121-c6bf-40b8-aa2f-069b8c25b763
# ╟─bf4b3a28-fbba-44f6-b4f1-751ec5658bfb
# ╟─d57e6f78-3da7-4bbd-a10e-d44b010025a8
# ╟─df7677f8-df25-48d4-8133-ffce71c65a48
# ╟─b6af72e6-07d8-4209-aa2e-829ad904a99c
# ╟─1d6fd6a8-1a40-4cb8-8f5b-af9b1f21ea36
# ╟─41ba6e50-747d-4f75-8d1d-718c7054baff
# ╟─06eb8148-ba1d-4b87-bb3a-57dc44be136f
# ╟─908d2e1a-64d4-4bbd-ad25-b8709de83a42
# ╟─4c7e30b8-5332-4bf3-a23b-4e5c49580ed4
# ╟─53a10970-7e9b-4bd4-bd65-b22028b66835
# ╟─73d51e22-3e40-4cbb-b649-65b719036647
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
