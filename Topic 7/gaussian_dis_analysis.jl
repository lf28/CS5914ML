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

# ╔═╡ 120a282a-91c1-11ec-346f-25d56e50d38c
begin
	using Distributions, StatsBase, Clustering
	using StatsPlots
	using PalmerPenguins, DataFrames
	using LogExpFunctions:logsumexp
	using Flux
end

# ╔═╡ 5aa0adbe-7b7f-49da-9bab-0a78108912fd
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

# ╔═╡ 22e1fbc9-f0bd-4159-b92f-11c412a660e6
using MLJLinearModels

# ╔═╡ 646dd3d8-6092-4435-aee9-01fa6a281bdc
ChooseDisplayMode()

# ╔═╡ 093c4c78-6179-4196-8d94-e548621df69b
TableOfContents()

# ╔═╡ 16497eaf-3593-45e0-8e6a-6783198663c3
md"""

# CS5914 Machine Learning Algorithms


#### Gaussian discriminant analysis
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ f7e989fd-d955-4323-bdef-57d9ffbe5a18
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true
	table = PalmerPenguins.load()
 	df = DataFrame(table)
end;

# ╔═╡ e136edb5-3e98-4355-83e2-55761eb8b15c
md"""
## Why probabilistic approach

A key message I hope to convey here
!!! correct ""
	**Probabilistic models** unify most *interesting* machine learning models 
    * supervised learning
	* and also **unsupervised learning**

In other words: **from probabilistic models' eyes, they are the same**


**Machine learning** are just **probabilistic inferences**:

$$P(y|x)$$

* assume different $P(\cdot)$ and plug in different $x$ and $y$ for different problems/situations
  * e.g. regression: $P$ is Gaussian;
  * and classification: $P$ is Bernoulli or Multinoulli 
* we will come back to this key message at the end of next lecture

""";

# ╔═╡ 2acf2c33-bd3b-4369-857a-714d0d1fc600
md"""

## Recap: Bayes' rule
"""

# ╔═╡ 6775b1df-f9dd-423e-a6ef-5c9b345e5f0f
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/bayes.png' width = '400' /></center>"

# ╔═╡ 76c05a93-698c-4ee1-9411-8a518c4e56b0
md"""

## Recap: Bayes' rule
"""

# ╔═╡ 21803313-fa45-48ea-bd3f-a30bc7696215
TwoColumn(md""" 
```math
\begin{align}
&\text{Hypothesis: }h \in \{\texttt{healthy}, \texttt{cold}, \texttt{cancer}\}\\
&\text{Observation: }\mathcal{D} = \{\texttt{cough} = \texttt{true}\}
\end{align}
```

Apply Bayes' rule

$$\begin{align}P(h|\texttt{cough}) &\propto P(h) P(\texttt{cough}|h) \\
&= \begin{cases} 0.321 & h=\texttt{healthy} \\
0.614 & h=\texttt{cold}\\ 
0.065 & h=\texttt{cancer}\end{cases}
\end{align}$$

$\hat{h}_{\rm MAP} =\arg\max_h p(h|\texttt{cough})=\texttt{cold}$

""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/cough_bn.svg" height = "210"/></center>
""")

# ╔═╡ 3a747ab4-8dc1-48ef-97ae-a736448020b3
md"""

## Recap: multi-variate Gaussian
"""

# ╔═╡ 501594f4-323f-4b99-93a4-8306e3228b63
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_md_alt.png' width = '900' /></center>"

# ╔═╡ f7829734-0b56-43a4-ab26-37b74af5cfd8
md"""

* the kernel is a **distance measure** between $x$ and $\mu$ (adjusted with correlations)

  $\large d_{\boldsymbol{\Sigma}}^2 = (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$

"""

# ╔═╡ 2496221f-70f9-4187-b329-35fbaf03a480
md"""

## Recap: multi-variate Gaussian


###### ``\mathbf\Sigma`` leads to different distance measures
"""

# ╔═╡ c05d44bd-1564-430a-b633-b57cca1f5526
Σs = [Matrix(1.0I, 2,2), [2 0; 0 0.5],  [.5 0; 0 2] , [1 0.9; 0.9 1], [1 -0.9; -0.9 1]];

# ╔═╡ fd0a8334-07e6-4678-bf28-d334d81fc67e
plts_mvns=let
	Random.seed!(123)
	nobs= 250
	plts = []

	for Σ in Σs
		mvn = MvNormal(zeros(2), Σ)
		data = rand(mvn, nobs)
	  	# scatter(data[1,:], data[2,:])
		plt = plot(-3:0.1:3, -3:0.1:3, (x, y) -> pdf(mvn, [x,y]), st=:contour, c=:jet, clabels=false, ratio=1, lw=2, levels=5, colorbar=false, framestyle=:origin)

		# plot!(-3:0.1:3, -3:0.1:3, (x, y) -> pdf(mvn, [x,y]), st=:contour, c=:jet, ratio=1, lw=3, levels=5, colorbar=false, framestyle=:origin)
		# scatter!(data[1,:], data[2,:], c=1, alpha=0.5, ms=2, label="")
		push!(plts, plt)
	end

	# color=:turbo, clabels=true,
	plts
end;

# ╔═╡ 9ecd708a-0585-4a68-b222-702e8de02abb
ThreeColumn(
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[1]))

$(plot(plts_mvns[1], size=(220,220)))
	


"""	,
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[2]))

$(plot(plts_mvns[2], size=(220,220)))
	



"""
	,


md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[3]))


$(plot(plts_mvns[3], size=(220,220)))
	


"""
)

# ╔═╡ 133e4cb5-9faf-49b8-96ef-c9fc5c8e0b94
TwoColumn(

md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[4]))



$(plot(plts_mvns[4], size=(320,220)))
	

"""
	,


md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[5]))

$(plot(plts_mvns[5], size=(320,220)))
	



"""
)

# ╔═╡ 2bed5f9e-3a01-4629-a92e-6ccca2d78119
md"""
## Recap: probabilistic discriminative models
##### (softmax regression)
\

Recall the probabilistic model for **softmax regression**, which is a **discriminative model**

> ```math
> \large
> p(y^{(i)}= c|\mathbf{W}, \mathbf{x}^{(i)}) = \texttt{softmax}_c(\mathbf{W}\mathbf{x}^{(i)} +\mathbf{b});\; \text{for }c= 1\ldots C
> ```


* ##### the input features ``\mathbf{x}^{(i)}`` are assumed fixed (*non-random*)

"""

# ╔═╡ 1a2c28b2-0821-4791-a52e-00b719cd1049
md"""
## Recap: probabilistic discriminative models
##### (softmax regression)
\

Recall the probabilistic model for **softmax regression**:

> ```math
> \large
> 
> p(y^{(i)}= c|\mathbf{W}, \mathbf{x}^{(i)}) = \texttt{softmax}_c(\mathbf{W}\mathbf{x}^{(i)} +\mathbf{b});\; \text{for }c= 1\ldots C
> ```


* ##### the input features ``\mathbf{x}^{(i)}`` are assumed fixed (*non-random*)
* ##### but ``y^{(i)}`` are *random* and is categorical distributed

$y^{(i)} \sim p(y^{(i)}|\mathbf{x})=\mathcal{Cat}(\texttt{softmax}(\mathbf{W}\mathbf{x}^{(i)} +\mathbf{b}))$
"""

# ╔═╡ cc9d9e19-c464-4ff7-ae1f-bba4111075b3
md"""
## Probabilistic generative models 

##### (Gaussian discriminant analysis (GDA)) 
\

In **contrast**, *probabilistic* **generative models** assume 

##### Both ``\{\mathbf{x}^{(i)},y^{(i)}\}`` are *random*
  * ###### ``\large y^{(i)}\in \{1,2,\ldots,C\}``, *i.e.* *Categorical* distributed
  * ###### ``\large \mathbf{x}^{(i)} \in \mathbb{R}^d`` *Gaussian* distributed  


"""

# ╔═╡ 200b1531-2a94-472a-9fd2-b90b04877581
md"""
## Probabilistic generative models

Very similar to the coughing example
* ``y^{(i)}\in \{1\ldots C\}``, the class label is considered the cause
* ``\mathbf{x}^{(i)} \in \mathbb{R}^d``, the measurements are considered the effect
"""

# ╔═╡ c4260ef4-d521-4d23-9495-41741940f128
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gda_bn.svg" height = "280"/></center>
"""

# ╔═╡ 52b5327c-db4b-4f58-951a-ffba6677abd6
md"""




## Probabilistic generative models  - GDA





"""

# ╔═╡ 8f61318f-6a39-45a6-a822-dc618b33c5cd
TwoColumn(md"""##### (Gaussian discriminant analysis) 
\

In contrast, **generative models** assume 

##### Both ``\mathbf{x}^{(i)}`` and ``y^{(i)}`` *random*
  * ###### ``\large y^{(i)}\in \{1,2,\ldots,C\}``, *i.e.* *Categorical* distributed
  * ###### ``\large \mathbf{x}^{(i)} \in \mathbb{R}^d`` (conditional on ``y^{(i)}``) *Gaussian* distributed """, html"""<br/><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gda_bnalone.svg" width = "250"/></center>
""")

# ╔═╡ 1b9c9ccd-5d90-4e26-b6f4-da47c4c58619
md"""

## Penguine data example
"""

# ╔═╡ bbfba1d5-c280-43e1-9721-d0ab2b9226ed
begin

	plt2 = @df df scatter(:bill_length_mm, :bill_depth_mm, group =:species, framestyle=:origins,  xlabel="bill length (mm)", ylabel="bill depth (g)");


end;

# ╔═╡ 46f18f6b-04c6-47fd-b50c-9db3f90c8f4b
TwoColumn(html"<br/><br/><center><img src='https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png
' width = '600' /></center>",  plot(plt2, size=(350,350)))

# ╔═╡ 9938af3d-0cad-4005-b2f6-db7fcf84b89a
md"""

## Penguine data example (cont.)
"""

# ╔═╡ f033fbb8-55b3-40a7-a2e1-db8777daffc6
md"""
*  ``y^{(i)} \in \{\text{Adelie}, \text{Chinstrap}, \text{Gentoo}\}`` is the **cause**
  * ``p(y)``: the prior distribution over the three species

*   ``\mathbf{x}^{(i)}``: the bill length and depth measurements of a penguine is the **effect** of ``y^{(i)}``
  * ``p(\mathbf{x}|y=\text{Chinstrap})``: a Gaussian likelihood (red cluster)
"""

# ╔═╡ eb7b07f3-6efe-4163-a7c5-33ff783c391b
md"""




## Probabilistic generative models - GDA

"""

# ╔═╡ 1ea85299-83c9-4c7e-8f90-7f37455b3db1
TwoColumn(md"""

##### To be more specific:


-----
for each pair ``i=1\ldots n``
1. generate ``y^{(i)}`` from its prior distribution
$y^{(i)}\sim \mathcal{Cat}(\boldsymbol\pi)=\begin{cases}\pi_1 & y^{(i)}=1 \\ \pi_2 & y^{(i)} =2 \\\vdots & \vdots \\ \pi_C & y^{(i)} =C\end{cases}$
2. generate observation ``\mathbf{x}^{(i)}`` conditional on ``y^{(i)}`` from the corresponding Gaussian
$\mathbf{x}^{(i)}|y^{(i)} \sim p(\mathbf{x} | y^{(i)}) = \mathcal{N}(\boldsymbol{\mu}_{y^{(i)}}, \boldsymbol{\Sigma}_{y^{(i)}})$
----
""", html"""<br/><br/><br/><br/><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gda_bnalone.svg" width = "250"/></center>
""")

# ╔═╡ cc9d8ae5-9ed3-407c-b63c-e8c4ea1cd472
md"""

## Demonstration
"""

# ╔═╡ 17c07bab-5d5a-4480-8e60-94ffc4d891ef
@bind init_reset Button("restart")

# ╔═╡ 9e21c264-1175-479f-a0bd-51b21c67ce36
md"""
start the simulation: $(@bind start CheckBox(default=false)),
$(@bind go Button("step"))

"""

# ╔═╡ 0ca6a1e1-6f91-42fa-84b5-7c3b9170e56a
begin
	gr()
	init_reset
	trueμs = [-3, 0, 3.]
	trueσs = [1 , 1, 1]
	trueπs = 1/3 * ones(3)
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	zs_samples = []
	# qda_plot = Plots.plot(lc=:gray, lw= .5, ls=:dash, xlim =[-6, 6], ylim =[0, 1.5], label="", framestyle=:zerolines, yaxis=false, title="")
	iters = [-1]
	ys_samples = []
end;

# ╔═╡ 60c2d8e3-a411-4811-bcbd-2bbf52ff4ac3
let
	gr()
	# logistic = σ
	Random.seed!(12345)
	n_obs = 30

	xs = sort(rand(n_obs) * 10 .- 5)
	logliks = hcat([logpdf.(mvns[k], xs) for k in 1:3]...)
	σs = exp.(logliks .- logsumexp(logliks, dims=2) )'
	ys = rand.([Categorical(p[:]) for p in eachcol(σs)])
	x_centre = 0
	py_x(x) = softmax(logpdf.(mvns, x))
	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:zerolines, labels=L"x", color=:black, ms=8, markershape=:x, xlabel=L"x", ylim=[-0.1, 1.2], ylabel=L"p(y|x)", legend=:outerbottom, size=(800,500))
		# # true_w =[0, 1]
	for c in 1:3
		plot!(plt, min(-6 + x_centre, -5):0.01:max(x_centre +6, 5), (x) -> py_x(x)[c], lw=2, lc=c, label="softmax "*L"\sigma_{%$(c)}(x) = p(y =%$(c)|x)", legendfontsize =15, title="Probabilistic discriminative model: "*L"{x}"* " are fixed (non-random)")
	end
	plt
end

# ╔═╡ 4c10e534-8f0e-4767-9267-90d67a251dde
let
	gr()
	# logistic = σ
	Random.seed!(12345)
	n_obs = 30

	xs = sort(rand(n_obs) * 10 .- 5)
	logliks = hcat([logpdf.(mvns[k], xs) for k in 1:3]...)
	σs = exp.(logliks .- logsumexp(logliks, dims=2) )'
	ys = rand.([Categorical(p[:]) for p in eachcol(σs)])
	ys = rand.([Categorical(p[:]) for p in eachcol(σs)])
	x_centre = 0


	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel=L"x", ylim=[-0.1, 1.2], ylabel=L"p(y|x)", legend=:outerbottom)
	py_x(x) = softmax(logpdf.(mvns, x))
		# # true_w =[0, 1]
	for c in 1:3
		plot!(plt, min(-6 + x_centre, -5):0.01:max(x_centre +6, 5), (x) -> py_x(x)[c], lw=2, lc=c, label="softmax: " * L"\sigma_{%$(c)}(x)=p(y=%$(c)|x)", title="Probabilistic discriminative model: "*L"y\sim p(y|x)")
	end

	xis = xs
	
	anim = @animate for i in 1:length(xis)
		x = xis[i]
		scatter!(plt, [xis[i]],[0], markershape = :circle, label="", c=ys[i], markersize=5)
		vline!(plt, [x], ls=:dash, lw=0.2, lc=:gray, label="")
		plt2 = plot(Categorical(σs[:, i]), st=:bar, yticks=(1:3, [L"y= 1", L"y =2", L"y= 3"]), xlim=[0,1.01], orientation=:h, yflip=true, label="", color=[1,2,3], title=L"p(y|{x})")
		plot(plt, plt2, layout=grid(2, 1, heights=[0.8, 0.2]),size=(700,500))
	end

	gif(anim; fps=4)
end

# ╔═╡ 9a0ea270-10d8-44f5-98a1-f6324572548e
TwoColumn(
	md"""

Assume a ``C=3``, *i.e.* three class problems, 

**The prior** (uniform prior)

$p(y) = \mathcal{Cat}(\boldsymbol\pi) = \begin{cases}\colorbox{lightblue}{1/3} & y=1\\ \colorbox{lightsalmon}{1/3} & y=2\\ \colorbox{lightgreen}{1/3} &y=3 \end{cases}$

**The likelihood**: ``p(\mathbf{x}|y)``, *i.e.* Gaussian components:

```math
\begin{align}
&\colorbox{lightblue}{$p(x|y=1) = \mathcal{N}_1(-3, 1)$}\; \\
&\colorbox{lightsalmon}{$p(x|y=2) = \mathcal{N}_2(0, 1)$}\; \\
&\colorbox{lightgreen}{$p(x|y=3) = \mathcal{N}_3(3, 1)$}
\end{align}
```
"""
	
	,let
	go
	if !isempty(zs_samples)
		i = length(zs_samples)
		zi = zs_samples[end]
		qda_plot =	plot((x) -> pdf(mvns[zi], x), fill=true, alpha=0.5, lc = zi , c=zi, lw= 2, label="sample "*L"y^{(%$i)}\sim \mathcal{Cat}(\pi);\;"*L" y^{(%$i)}=%$(zi)",  xlim =[-6, 6], framestyle=:zerolines, yaxis=false, title="", size=(300,350))
		for k in 1:3
			plot!((x) -> pdf(mvns[k], x), lc=k, lw= 1, fill=false, ls=:dash, label="")
		end

		scatter!([ys_samples[end]], [0], markershape=:x, markerstrokewidth=5, c= zi, ms=8, label="sample "*L"x^{(%$i)}\sim \mathcal{N}_{%$(zi)}", legendfontsize=12, legend=:outertop)
			
		for c_ in 1:3
			zs_ = zs_samples .== c_
			nc = sum(zs_)
			if nc > 0
				scatter!(ys_samples[zs_], zeros(nc), markershape=:diamond, ms=6, c= c_, alpha=0.5, label="")
			end
		end
	else
		qda_plot =	plot((x) -> pdf(mvns[1], x), lc=1, lw= .5, fill=true, c=1, alpha=0.5, ls=:dash, xlim =[-6, 6], label="", framestyle=:zerolines, yaxis=false, title="Probabilistic generative model (3 classes)", titlefontsize=8, size=(300,350))
		plot!((x) -> pdf(mvns[2], x),lc=2, lw=.5,fill=true, c=2, alpha=0.5,  ls=:dash, label="")
		plot!((x) -> pdf(mvns[3], x), lc=3, lw=.5, fill=true, c=3, alpha=0.5, ls=:dash, label="")
	end
	qda_plot
end)

# ╔═╡ e0473be1-1ee0-42fe-95a1-cdd6c948fb35
begin
	go
	if start 
		iters[1] += 1
		zi = sample(Weights(trueπs))
		push!(zs_samples, zi)
		di = rand(mvns[zi])
		push!(ys_samples, di)
		zi, di
	end
end;

# ╔═╡ 77aaff69-13bb-4ffd-ad63-62993e13f873
md"""

## An one-dimensional example 
##### (generative model)


\


----
```math
\begin{align}
y \;&\;\sim\; \mathcal{Cat}(\boldsymbol\pi) \tag{generate $y$}\\
x|y=c \; &\;\sim\; \mathcal{N}_c(\mu_c, \sigma^2_c) \tag{generate $x$ conditional on $y$}
\end{align}
```
----

* ``c=1\ldots 3``, *i.e.* three-class classification

"""

# ╔═╡ e5a23ba6-7859-4212-8854-86b238332eef
let
	# data_d1 = data₁[:, 1]
	nn = 30
	trueμs = [-3, 0, 3.]
	trueσs = [1 , 1, 1]
	# trueπs = [0.15, 0.7, 0.15]
	trueπs = 1/3 * ones(3)
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	truezs = zeros(Int, nn)
	data = zeros(nn)
	Random.seed!(123)
	anim_gen = @animate for i in 1:nn
		truezs[i] = zi = sample(Weights(trueπs))
		plot((x) -> pdf(mvns[1], x), lc=:gray, lw= .5, ls=:dash, xlim =[-6, 6], label="", framestyle=:zerolines, yaxis=false, title="")
			# title="Sample "*L"y^{(%$i)}\sim \mathcal{Cat}(%$(round.(trueπs, digits=2)))\;" * "; "*L"y^{(%$(i))} = %$(zi)")
		plot!((x) -> pdf(mvns[2], x),lc=:gray, lw=.5, ls=:dash, label="")
		plot!((x) -> pdf(mvns[3], x), lc=:gray, lw=.5, ls=:dash, label="")
		# ci = zi == 1 ? :gray : 1 
		plot!((x) -> pdf(mvns[zi], x), fill=true, alpha=0.5, lc = zi , c=zi, lw= 2, label="")

		data[i] = di = rand(mvns[zi])
		scatter!([di], [0], markershape=:x, markerstrokewidth=5, c= zi, ms=8, label="sample "*L"y^{(%$i)}\sim \mathcal{Cat}(\pi)\;" * "; "*"sample "*L"x^{(%$i)}\sim \mathcal{N}_{%$(zi)}", legendfontsize=12, legend=:outertop)
		
		for c_ in 1:3
			zs_ = truezs[1:i-1] .== c_
			nc = sum(zs_)
			if nc > 0
				scatter!(data[1:i-1][zs_], zeros(nc), markershape=:diamond, ms=6, c= c_, alpha=0.5, label="")
			end
		end
	end

	gif(anim_gen, fps=1)
end

# ╔═╡ 85eeec0d-f69f-4cf9-a718-df255a948433
WW = let
	nn = 100
	trueμs = [-3, 0, 3.]
	trueσs = [1 , 1, 1]
	# trueπs = 1/3 * ones(3)
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	labels = Int[]
	data = Float64[]
	for k in 1:3
		append!(data, rand(mvns[k], nn))
		append!(labels, k * ones(Int, nn))
	end
	λ = 0.0
	mlr = MultinomialRegression(λ) # you can also just use LogisticRegression
	theta = MLJLinearModels.fit(mlr, Matrix(reshape(data, :, 1)), labels)
	p = 2
	c = 3
	W = reshape(theta, p, c)
end;

# ╔═╡ 1ee13166-f4cb-46e1-83c0-5365ca707b9a
md"""

## A multi-variate example

The same idea, but we sample from a **multi-variate Gaussian**

----
```math
\begin{align}
y \;&\;\sim\; \mathcal{Cat}(\boldsymbol\pi) \tag{generate $y= 1\ldots 3$}\\
x|y=c \; &\;\sim\; \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c) \tag{generate $x$ conditional on $y$}
\end{align}
```
----

* ``c=1\ldots 3``, *i.e.* three-class classification

"""

# ╔═╡ 6b27d9f7-e74c-404a-8248-c67cbbfe8342
let
	# data_d1 = data₁[:, 1]
	nn = 30
	trueμs = [[-3.0, 2.0], [3.0, 2.0], [0., -2]]
	trueπs = [0.15, 0.7, 0.15]
	trueπs = 1/3 * ones(3)
	# mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	# truezs = zeros(Int, nn)
	data = zeros(2, nn)
	Random.seed!(123)
	mvns = [MvNormal(trueμs[k], 1) for k in 1:3]
	truezs = rand(Categorical(trueπs), nn)
	anim_gen = @animate for (i, zi) in enumerate(truezs)
		plot(-7:0.1:7, -6:0.1:6, (x, y) -> pdf(mvns[zi], [x, y]), st=:surface, alpha=0.25, label="", c=zi, colorbar=false, framestyle=:none, size=(500,500), zaxis=false)
		data[:, i] = di = rand(mvns[zi])
		scatter!([di[1]], [di[2]], [0],markershape=:x, markerstrokewidth=5, c= zi, ms=8, label="sample "*L"y^{(%$i)}\sim \mathcal{Cat}(\pi)\;" * "; "*"sample "*L"x^{(%$i)}\sim \mathcal{N}_{%$(zi)}", legendfontsize=12, legend=:top)
		
		for c_ in 1:3
			zs_ = findall(truezs[1:i-1] .== c_)
			nc = length(zs_)
			if nc > 0
				scatter!(data[1, zs_], data[2, zs_], zeros(nc), markershape=:circle, ms=4, c= c_, alpha=0.5, label="")
			end
		end
	# 	# plot!((x) -> trueπs[2] * pdf(mvns[2], x),lc=(zi == 2 ? 2 : :gray), lw= (zi == 2 ? 2 : 1), label="")
	# 	# plot!((x) -> trueπs[3] * pdf(mvns[3], x),lc=(zi == 3 ? 3 : :gray), lw= (zi == 3 ? 2 : 1), label="")

	end

	# # truezs
	
	# # density(data_d1)
	gif(anim_gen, fps=1)
end

# ╔═╡ 46d30b0c-0cf5-4369-a81f-6c234043a8ea
md"""

## Why generative model ?

!!! note "Discriminative model"
	```math
	\large p(y|\mathbf{x})
	```

!!! note "Generative model"
	```math
	\large p(y, \mathbf{x}) = p(y) p(\mathbf{x}|y)
	```


* note that ``\large p(\mathbf{x}|y)`` is unique to the generative model

##### Applications of the generative model ``\large p(\mathbf{x}, y)``:

1. outlier detection; 

```math
\mathbf{x}\text{ is an outlier if: }\;P(\mathbf{x}|y) < \epsilon; \; \text{where }\epsilon \text{ is a small constant}
```
2. deal with missing data 

```math
\mathbf{x}=[\texttt{missing}, x_2]^\top;\;p(\mathbf{x}|y) = \sum_{x_1} p(X_1= x_1, x_2|y)
```
3. simulate pseudo/fake data
```math
\mathbf{x}\sim p(\mathbf{x}|y=c)\;\; \text{or}\;\; \mathbf{x}\sim p(\mathbf{x})=\sum_{c=1}^C p(y=c)p(\mathbf{x}|y=c)
```
"""

# ╔═╡ abd46485-03c9-4077-8ace-34f9b897bb04
md"""

## Classification rule (Bayes' rule)

To classify ``y`` given ``\mathbf{x}``, apply **Bayes' rule**,


```math
\large
p({y}|\mathbf{x}) = \frac{p(y) p(\mathbf{x}|y)}{p(\mathbf{x})}
```


* ``p(y=c) = \pi_c``, the prior 

* ``p(\mathbf{x}|y=c) = \mathcal{N}_c(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)``, Gaussian likelihood

Note that the normalising constant is defined as 

```math
\large
p(\mathbf{x}) = \sum_{c} p(y=c) p(\mathbf{x}|y=c)
```

Therefore, 

```math
\large
p({y}=c|\mathbf{x}) = \frac{\pi_c\cdot \mathcal{N}(x; \mu_c, \Sigma_c)}{\sum_c \pi_c\cdot \mathcal{N}(x; \mu_c, \Sigma_c)}
```
"""

# ╔═╡ dd915c43-b3cc-4608-87b0-852b2d533b15
md"""


## Demonstration

Note that 

* ``p(y|x)`` is always sum to one; *i.e.* a valid posterior probability vector

"""

# ╔═╡ 0bab5178-5cbd-467e-959d-1f09b496d2af
md"Select ``x``: $(@bind x00 Slider(-4.5:0.1:4.5, default=0, show_value=true))"

# ╔═╡ b10c879d-98c8-4617-91de-fba1d9499ba4
let
	# data_d1 = data₁[:, 1]
	nn = 30
	trueμs = [-1.5, 0, 1.5]
	trueσs = [1 , 1, 1]
	# trueπs = [0.15, 0.7, 0.15]
	trueπs = 1/3 * ones(3)
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	truezs = zeros(Int, nn)
	data = zeros(nn)
	Random.seed!(123)

	zs = rand(Categorical(trueπs), nn)
	# x0 =-3
	py = trueπs .* pdf.(mvns, x00)
	y0 = argmax(py)
	plt = plot(xlim =[-6, 6], label="", framestyle=:origins, yaxis=true, ylabel=L"p(y,x)", title="", legend=:outerbottom)
	for c in 1:3
		alpha = 0.4
		lw = 2
		label = L"{π_{%$c} \mathcal{N}(x; \mu_{%$c}, \sigma^2_{%$c})= %$(round(py[c];digits=3))}"
		plot!((x) -> trueπs[c] * pdf(mvns[c], x), lc=c, fill=true, alpha=alpha, lw=0.5, c=c, ls=:dash, label=label, legendfontsize = 12)
		# if showall
		plot!([x00], [py[c]], label="", c=:gray, lw=lw, alpha=1, lc=c, ls=:dash, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		# else
			# plot!([x0], [py[class]], label="", c=:gray, lw=lw, alpha=1, lc=class, ls=:solid, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		# end
	end
	py_ = py/sum(py)
	
	scatter!([x00], [0], label="", c=:gray, markershape=:x, markerstrokewidth=3, markersize=6,  title=L"p(y|x) = %$(round.(py_;digits=2))")


	# end

	plt
	
end

# ╔═╡ 57c4e40c-6760-4c88-896d-7ea8faf134e0
md"""

## Classification rule (MAP estimator)

When computing the **MAP** estimator, we do not really need to compute the normalising constant

```math
\large
p(\mathbf{x}) = \sum_{c=1}^C p(y=c) p(\mathbf{x}|y=c)
```
* it can be **expensive** to compute

*Instead*, **since**

$\large p(y=c|\mathbf{x}) \propto p(y=c) p(\mathbf{x} |y=c)$



**Therefore**,

```math
\large
\arg\max_c \;p(y=c|\mathbf{x}) = \arg\max_c\; \{p(y=c) p(\mathbf{x} |y=c)\}
```


> To make the prediction ``\hat{y} \in \{1\ldots C\}``:
> ```math
> \large
> \begin{align}
> \hat{y} 
> &= \arg\max_c\; \pi_c \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c,  \boldsymbol{\Sigma}_c)
> \end{align}
> ```


"""

# ╔═╡ 1a6cb45d-1b26-4efa-bd40-f7a8e3bbd770
md"""
## Demonstration
"""

# ╔═╡ c85b688c-fc8d-4dfa-98bd-9e43dd0b79d5
md"Select ``x``: $(@bind x0 Slider(-5.5:0.1:5.5, default=0, show_value=true))"

# ╔═╡ ee10a243-052f-4c0f-8f0d-e16ad6ceb611
md"Select ``y``: $(@bind class Select(1:3)),
Show all ``y``: $(@bind showall CheckBox(default=false)) ,
Add descision boundary $(@bind add_db CheckBox(default=false))
"

# ╔═╡ df0719cb-fc54-456d-aac7-48237a96cbdd
let
	# data_d1 = data₁[:, 1]
	nn = 30
	trueμs = [-1.5, 0, 1.5]
	trueσs = [1 , 1, 1]
	# trueπs = [0.15, 0.7, 0.15]
	trueπs = 1/3 * ones(3)
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	truezs = zeros(Int, nn)
	data = zeros(nn)
	Random.seed!(123)

	zs = rand(Categorical(trueπs), nn)
	x1_ = (trueμs[1:2]) |> mean
	x2_ = (trueμs[2:end]) |> mean
	# x0 =-3
	py = trueπs .* pdf.(mvns, x0)
	y0 = argmax(py)
	plt = plot(xlim =[-6, 6], label="", framestyle=:origins, yaxis=true, ylabel=L"p(y,x)", title="", legend=:outerbottom)
			# title="Sample "*L"y^{(%$i)}\sim \mathcal{Cat}(%$(round.(trueπs, digits=2)))\;" * "; "*L"y^{(%$(i))} = %$(zi)")
	for c in 1:3
		alpha = c == class ? 0.5 : 0.05
		lw=2
		if showall
			alpha = 0.25
			lw = 2
		end
		label = L"{π_{%$c} \mathcal{N}(x; \mu_{%$c}, \sigma^2_{%$c})= %$(round(py[c];digits=3))}"
		plot!((x) -> trueπs[1] * pdf(mvns[c], x), lc=c, fill=true, alpha=alpha, lw=0.5, c=c, ls=:dash, label=label, legendfontsize = 12)
		if showall
			plot!([x0], [py[c]], label="", c=:gray, lw=c, alpha=1, lc=c, ls=:dash, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		else
			plot!([x0], [py[class]], label="", c=:gray, lw=lw, alpha=1, lc=class, ls=:solid, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		end
	end
	
	scatter!([x0], [0], label="", c=:gray, markershape=:x, markerstrokewidth=3, markersize=6,  title=L"\hat{y} = \arg\max_c\; \{\pi \cdot \mathcal{N}(x)\}=%$(y0)")


	if add_db
		plot!([x1_], [trueπs[1] * pdf(mvns[1], x1_)], label="", lc=1, lw=2.5, alpha=1,  ls=:dash, markersize=1, st=:sticks)
		plot!([x2_], [trueπs[2] * pdf(mvns[2], x2_)], label="", lc=2, lw=2.5, alpha=1, ls=:dash, markersize=1, st=:sticks)
	end

	# end

	plt
	
	# plot!((x) -> pdf(mvns[3], x), lc=:gray, lw=.5, ls=:dash, label="")
	# anim_gen = @animate for i in 1:nn
	# 	truezs[i] = zi = sample(Weights(trueπs))
	# 	plot((x) -> pdf(mvns[1], x), lc=:gray, lw= .5, ls=:dash, xlim =[-6, 6], label="", framestyle=:zerolines, yaxis=false, title="")
	# 		# title="Sample "*L"y^{(%$i)}\sim \mathcal{Cat}(%$(round.(trueπs, digits=2)))\;" * "; "*L"y^{(%$(i))} = %$(zi)")
	# 	plot!((x) -> pdf(mvns[2], x),lc=:gray, lw=.5, ls=:dash, label="")
	# 	plot!((x) -> pdf(mvns[3], x), lc=:gray, lw=.5, ls=:dash, label="")
	# 	# ci = zi == 1 ? :gray : 1 
	# 	plot!((x) -> pdf(mvns[zi], x), fill=true, alpha=0.5, lc = zi , c=zi, lw= 2, label="")

	# 	data[i] = di = rand(mvns[zi])
	# 	scatter!([di], [0], markershape=:x, markerstrokewidth=5, c= zi, ms=8, label="sample "*L"y^{(%$i)}\sim \mathcal{Cat}(\pi)\;" * "; "*"sample "*L"x^{(%$i)}\sim \mathcal{N}_{%$(zi)}", legendfontsize=12, legend=:outertop)
		
	# 	for c_ in 1:3
	# 		zs_ = truezs[1:i-1] .== c_
	# 		nc = sum(zs_)
	# 		if nc > 0
	# 			scatter!(data[1:i-1][zs_], zeros(nc), markershape=:diamond, ms=6, c= c_, alpha=0.5, label="")
	# 		end
	# 	end
	# end

	# gif(anim_gen, fps=1)
end

# ╔═╡ b4d619a1-8741-4902-86f8-cd8e84c9d785
md"""

## Effect of prior ``p(y) = \boldsymbol{\pi}``
"""

# ╔═╡ 2ad600a2-4e5d-4af6-a18c-caaa516a542d
begin
	trueπss_ = [[1,1,1],[1, 4, 1], [4, 1, 1], [1, 1, 4]]
	trueπss_ = [trueπss_[k]/sum(pi) for (k, pi) in enumerate(trueπss_)]
end;

# ╔═╡ e243fe55-ee3e-47dc-9a7a-4319e0e86f8e
@bind trueπs_ Select(trueπss_)

# ╔═╡ c63369ed-58ed-4dd3-9292-c6c265ad52ba
md"Select ``x``: $(@bind x0_ Slider(-5.5:0.1:5.5, default=0, show_value=true))"

# ╔═╡ 30cf6d78-c541-4d2d-b455-cb365d52b5cd
let
	# data_d1 = data₁[:, 1]
	nn = 30
	trueμs = [-1.5, 0, 1.5]
	trueσs = [1 , 1, 1]
	# trueπs = [0.15, 0.7, 0.15]
	trueπs = trueπs_
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	truezs = zeros(Int, nn)
	data = zeros(nn)
	Random.seed!(123)

	zs = rand(Categorical(trueπs), nn)
	x0 = x0_
	py = trueπs .* pdf.(mvns, x0)
	y0 = argmax(py)
	plt = plot(xlim =[-6, 6], label="", framestyle=:origins, yaxis=true, ylabel=L"p(y,x)", title="", legend=:outerbottom)
			# title="Sample "*L"y^{(%$i)}\sim \mathcal{Cat}(%$(round.(trueπs, digits=2)))\;" * "; "*L"y^{(%$(i))} = %$(zi)")
	for c in 1:3
		alpha =  c == y0 ? 0.8 : 0.2 
		lw=2
		label = L"{π_{%$c} \mathcal{N}(x; \mu_{%$c}, \sigma^2_{%$c})= %$(round(py[c];digits=3))}"
		plot!((x) -> trueπs[c] * pdf(mvns[c], x), lc=c, fill=true, alpha=alpha, lw=0.5, c=c, ls=:dash, label=label, legendfontsize = 12)
		# if showall
		plot!([x0], [py[c]], label="", c=:gray, lw=c, alpha=1, lc=c, ls=:dash, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		# else
		# 	plot!([x0], [py[class]], label="", c=:gray, lw=lw, alpha=1, lc=class, ls=:solid, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		# end
	end
	
	scatter!([x0], [0], label="", c=:gray, markershape=:x, markerstrokewidth=3, markersize=6,  title=L"\hat{y} = \arg\max_c\; \pi \cdot \mathcal{N}(x)=%$(y0)")

	j, k = 1, 2
	x1_ = 0.5 * (trueμs[j] + trueμs[k]) - (log(trueπs[j]) -log(trueπs[k]))/(trueμs[j] - trueμs[k])
	j, k = 2, 3
	x2_ = 0.5 * (trueμs[j] + trueμs[k]) - (log(trueπs[j]) -log(trueπs[k]))/(trueμs[j] - trueμs[k])
	# if add_db
	plot!([x1_], [trueπs[1] * pdf(mvns[1], x1_)], label="", lc=1, lw=3, alpha=1, ls=:dash, markersize=1, st=:sticks)
	plot!([x2_], [trueπs[2] * pdf(mvns[2], x2_)], label="", lc=2, lw=3, alpha=1, ls=:dash, markersize=1, st=:sticks)
	# end

	plt
	# x1_
	
end

# ╔═╡ e33b07c9-fb73-405d-8ee0-6e6e88e32bab
md"""

## Classification in practice (use log)


In practice, to find **MAP**, we apply the ``\ln``-transform 

```math
\large
\begin{align}
\hat{y} &= \arg\max_c\; \pi_c \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c) \\
&= \arg\max_c\; \ln \pi_c + \ln  \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)
\end{align}
```

Sub-in Gaussian's likelihood we have


```math
\large
\begin{align}
\hat{y} 
&= \arg\max_c\, \ln \pi_c + \ln  \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c) \\
&= \arg\max_c\, \left \{\ln \pi_c - \frac{1}{2}\ln |\boldsymbol\Sigma_c|-\frac{1}{2}(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}_c^{-1}(\mathbf{x} -\boldsymbol{\mu}_c) \underbrace{\cancel{- \frac{d}{2}\ln 2\pi}}_{\rm constant} \right \}
\end{align}
```


"""

# ╔═╡ c363c7d2-5749-49cb-8643-57a1b9dda8eb
md"""

## Linear discriminant analysis (LDA)


> **Linear discriminant analysis (LDA)** model assumes for all ``c``, 
> 
> $\large \large \boldsymbol{\Sigma}_c = \boldsymbol{\Sigma}\;\; \text{for all }c=1\ldots C$

Then the decision rule reduces to

```math
\large
\begin{align}
\hat{y} 
&= \arg\max_c\, \left \{\ln \pi_c - \underbrace{\cancel{\frac{1}{2}\ln |\boldsymbol\Sigma|}}_{\rm constant}-\frac{1}{2}(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c) \right \} \\


&= \arg\max_c\, \left \{\ln \pi_c -\frac{1}{2}(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c) \right \}
\end{align}
```


"""

# ╔═╡ 51b1572f-5965-41a8-b6d6-061c48f9af0c
md"""

## Linear discriminant analysis (LDA)


**Linear discriminant analysis (LDA)** model assumes for all ``c``, 


$\large \large \boldsymbol{\Sigma}_c = \boldsymbol{\Sigma}\;\; \text{for all }c=1\ldots C$

Then the decision rule reduces to

```math
\large
\begin{align}
\hat{y} 
&= \arg\max_c\, \left \{\ln \pi_c - \underbrace{\cancel{\frac{1}{2}\ln |\boldsymbol\Sigma|}}_{\rm constant}-\frac{1}{2}(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c) \right \} \\


&= \arg\max_c\, \left \{\ln \pi_c -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c)$} }_{\text{distance btw } \mathbf{x} \text{ and } \boldsymbol{\mu}_c}\right \}
\end{align}
```

* ###### *i.e.* classify ``y`` based on its distance to the centers (adjusted with the prior)

* the decision boundary is therefore **linear**
"""

# ╔═╡ 89bfebb4-ee0d-46c8-956a-6a5088599ae6
md"""

## * Why linear?


!!! question "Question"
	Why the decision boundary is linear ?

The **decision boundary** between class 1 and 2 is



```math
\ln \pi_1 -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_1)$} }_{\text{distance btw } \mathbf{x} \text{ and } \boldsymbol{\mu}_1}= \ln \pi_2 -\frac{1}{2}\underbrace{\colorbox{lightpink}{$(\mathbf{x} -\boldsymbol{\mu}_2)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_2)$} }_{\text{distance btw } \mathbf{x} \text{ and } \boldsymbol{\mu}_2}
```

Multiply ``2`` on both sides,
```math
\begin{align}
2\ln \pi_1 -(\mathbf{x} -\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_1)= 2\ln \pi_2 -(\mathbf{x} -\boldsymbol{\mu}_2)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_2) \tag{$\times 2$}
\end{align}
```

Then expand the quadratic terms,

```math
\begin{align}
2\ln \pi_1 -(\cancel{\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x} }-&2\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\mathbf{x} + \boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1) = \\
&2\ln \pi_2 -(\cancel{\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}} -2\boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\mathbf{x} + \boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2) 
\end{align}

```

```math
\begin{align}
&\Rightarrow\;\; 2\ln \pi_1 + 2\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}-\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 = 2\ln \pi_2 + 2\boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}-\boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2\\
&\Rightarrow\;\; \underbrace{2(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)^\top\boldsymbol{\Sigma}^{-1}}_{\mathbf{w}^\top}\,\mathbf{x} = \underbrace{2\ln \frac{\pi_2}{\pi_1} +\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1-\boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2 }_{w_0}
\end{align}
```

Therefore, we have the following form

```math
\mathbf{w}^\top \mathbf{x} - w_0 =0
```
* which is a hyper-plane (linear function)
"""

# ╔═╡ eb14fb68-6950-4374-adef-35b583bf99fb
md"""

## LDA example

The covariance *e.g.* is identity matrix
```math
\large\boldsymbol{\Sigma}_c = \mathbf{I}
```
Then, the decision rule reduces to 

```math
\large
\begin{align}
\hat{y} 
&= \arg\max_c\, \left \{\ln \pi_c -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_c)^\top(\mathbf{x} -\boldsymbol{\mu}_c)$} }_{\text{Eucliean distance } \mathbf{x} \text{ and } \boldsymbol{\mu}_c}\right \}
\end{align}
```
"""

# ╔═╡ 2c3f2b50-0a95-4577-add8-8bf72580a44f
# let
# 	Random.seed!(123)
# 	K₁ =3
# 	n₁ = 600
# 	# D₁ = zeros(n₁, 2)
# 	# 200 per cluster
# 	truezs₁ = repeat(1:K₁; inner=200)
# 	trueμs₁ = zeros(2, K₁)
# 	trueμs₁[:,1] = [2.0, 2.0]
# 	trueμs₁[:,2] = [-2.0, 2]
# 	trueμs₁[:,3] = [0., -1.5]
# 	LL = cholesky([1 0; 0. 1]).L
# 	data₁ = trueμs₁[:,1]' .+ randn(200, 2) * LL
# 	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(200, 2)* LL)
# 	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(200, 2)* LL)
# 	plt₁ = plot(ratio=1, framestyle=:origin)
# 	truemvns = [MvNormal(trueμs₁[:, k], Matrix(I,2,2)) for k in 1:K₁]

# 	xs_qda = minimum(data₁[:,1])-0.1:0.1:maximum(data₁[:,1])+0.1
# 	ys_qda = minimum(data₁[:,2])-0.1:0.1:maximum(data₁[:,2])+0.1
# 	for k in 1:K₁
# 		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], c=k, label="Class"*string(k), alpha=0.8) 
# 		contour!(xs_qda, ys_qda, (x,y)-> pdf(truemvns[k], [x,y]), levels=5, colorbar = false, lw=2, alpha=0.7, c=:jet) 
# 	end
# 	plt₁
# end

# ╔═╡ ec66c0b1-f75c-41e5-91b9-b6a358cd9c3c
md"""

## LDA example

The covariance *e.g* is identity matrix
```math
\large\boldsymbol{\Sigma}_c = \mathbf{I}
```
Then, the decision rule reduces to 

```math
\large
\begin{align}
\hat{y} 
&= \arg\max_c\, \left \{\ln \pi_c -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_c)^\top(\mathbf{x} -\boldsymbol{\mu}_c)$} }_{\text{Eucliean distance } \mathbf{x} \text{ and } \boldsymbol{\mu}_c}\right \}
\end{align}
```
* assign ``\hat{y}`` based on the Euclidean distances to the centres
* note the decision boundary is linear !
"""

# ╔═╡ e28dda3f-fbcd-47f0-8f99-4f3a2087905d
md"Add boundary: $(@bind add_db_1 CheckBox(default=false))"

# ╔═╡ ea4a783e-6a18-4d4e-b0a2-bf4fd8070c7a
md"""

## More LDA example

The covariance *e.g* is tied but full
```math
\large\boldsymbol{\Sigma}_c = \begin{bmatrix}1 & 0.8 \\\ 0.8 & 1.0 \end{bmatrix}
```
Then, the decision boundary reduces to 

```math
\large
\begin{align}
\hat{y} 
&= \arg\max_c\, \left \{\ln \pi_c -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_c)^\top\mathbf{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c)$} }_{\text{mahalanobis distance } \mathbf{x} \text{ and } \boldsymbol{\mu}_c}\right \}
\end{align}
```
"""

# ╔═╡ 3b937208-c0bb-42e2-99a2-533113a0d4e0
md"Add boundary: $(@bind add_db_2 CheckBox(default=false))"

# ╔═╡ 1747e403-c5d6-471a-9f6e-fcf7ee8063d1
# let
# 	class = 3
# 	gr()
# 	data = data₁
# 	# K = K₃
# 	zs = truezs₁
# 	μs, Σ, πs = LDA_fit(data, zs)
# 	mvns = [MvNormal(μs[:, k], Symmetric(Σ)) for k in 1:size(μs)[2]]
# 	# mvns = []
# 	plt=plot(-7:.05:7, -6:0.05:6, (x,y) -> e_step([x, y]', mvns, πs)[1][:] |> argmax, c=:black, lw=1, alpha=0.7, title="Decision boundary by LDA", st=:contour, colorbar=false, ratio=1, framestyle=:grid)
# 	for k in [1,3,2]
# 		if k ==1
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		else
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		end
# 	end
# 	plt
# end

# ╔═╡ bcc88b1b-4e37-4d68-8af2-9b3923634dfd
md"""

## Quadratic discriminant analysis (QDA)


> **Quadratic discriminant analysis (QDA)** model assumes for all ``c``, 
> 
> $\large \large \boldsymbol{\Sigma}_c$
> are **not** necessarily the same

Then the decision rule becomes

```math
\large
\begin{align}
\hat{y} 

&= \arg\max_c\, \left \{\ln \pi_c- {{\frac{1}{2}\ln |\boldsymbol\Sigma_c|}} -\frac{1}{2}(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}_c^{-1}(\mathbf{x} -\boldsymbol{\mu}_c) \right \}
\end{align}
```

* ###### the decision boundary is *quadratic*

"""

# ╔═╡ 7c19c8ca-e30b-4214-b6f4-ac9eeb7d1a35
md"""

## * Why quadratic?


!!! question "Question"
	Why the QDA's decision boundary is quadratic ?


* left as an exercise


!!! hint "Hint"
	Show that the pair-wise decision boundary is a quadratic function, *i.e.*
	```math
	\mathbf{x}^\top \mathbf{A}\mathbf{x}+ \mathbf{w}^\top \mathbf{x}+ c =0
	```

"""

# ╔═╡ 853de250-143e-4add-b50d-2c73d1bc7910
md"""

## Example of QDA

Where assume the true parameters are known:

"""

# ╔═╡ cc60e789-02d1-4944-8ad5-718ede99669c
begin
	K₂ = 3
	trueμs₂ = zeros(2,K₂)
	trueΣs₂ = zeros(2,2,K₂)
	trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,2], trueΣs₂[:,:,2] = [2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,3], trueΣs₂[:,:,3] = [0., 0],  Matrix([0.5 0; 0 2])
	trueπs₂ = [0.2, 0.2, 0.6]
	truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
	n₂= 800
	truezs₂ = rand(Categorical(trueπs₂), n₂)
	data₂= vcat([rand(truemvns₂[z])' for z in truezs₂]...)
	# data₂, truezs₂ = sampleMixGaussian(n₂, truemvns₂, trueπs₂)
end;

# ╔═╡ ca168509-f902-4f01-a976-c0c0959b73f3
md"""

## More QDA example


We can investigate the posterior for a particular class ``c``, *e.g.*``c=3``


```math
\large
p(y=c|\mathbf{x}) \propto \pi_c \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)
```

* the correponding contour plot is
"""

# ╔═╡ c91d2a54-acf8-488b-b4c9-c1281a76e237
md"""

## More QDA example

"""

# ╔═╡ e5e0f46c-c0c3-4347-bd8f-33e2cd033cc8
md"""

## More QDA example


"""

# ╔═╡ 84a4326b-2482-4b5f-9801-1a09d8d20f5b
md"""

## More QDA example


We can investigate the posterior for a particular class ``c``


```math
\large
p(y=c|\mathbf{x}) \propto \pi_c \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)
```

* the correponding contour plot is
"""

# ╔═╡ 716ff72f-2299-472c-84b8-17315e8edc48
md"Choose class $(@bind class_d3 Select(1:3))"

# ╔═╡ 6094abc7-96bb-4c3f-9156-b5de5d7873f6
md"""

## Naive Bayes


##### Naive Bayes assumption: ``\mathbf\Sigma``s are all _diagonal_
* recall that diagonal ``\mathbf\Sigma`` ``\Leftrightarrow`` _Independence_ 
* axis aligned ellipsoids distance measures
"""

# ╔═╡ d4acae8a-8f27-4a2a-93d5-63d6b6b6db20
ThreeColumn(
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[1]))

$(plot(plts_mvns[1], size=(220,220)))
	


"""	,
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[2]))

$(plot(plts_mvns[2], size=(220,220)))
	



"""
	,


md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[3]))


$(plot(plts_mvns[3], size=(220,220)))
	


"""
)

# ╔═╡ 8ca2a156-e0f8-453f-a02c-daf23681bf89
md"""


###### Mathematically,  

**Naïve Bayes** implies **conditional independent** features of ``\mathbf{x}``, *i.e.*

> ```math
> \large 
> \begin{align}
> p(\mathbf{x}|y) &= p(x_1|y)p(x_2|y) \ldots p(x_d|y)\\
> &= \prod_{i=1}^d p(x_i|y)
> \end{align}
> ```



**Naïve Bayes** in a probabilistic graphical model (PGM) representation:
"""

# ╔═╡ d29b4fcb-d738-46e1-8f7d-e2df5913a067
html"""<br/><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/nb_bn_p3.svg" width = "950"/></center>
"""

# ╔═╡ 552da9d4-5ab9-427d-9287-feb1eca57183
md"""


## Model assumptions


It can be a disaster if we apply **LDA** on data with non-linear decision boundaries

* or when the co-variances are **very different** (the **QDA** assumption)
"""

# ╔═╡ 55f2aacf-a342-4d4c-a1b7-60c5c29ab340
md"Highlight mis-classified instances: $(@bind add_mis CheckBox(default=false))"

# ╔═╡ 5a963c3d-46cd-4697-8991-5d5e1bb9a9e5
# let
# 	gr()
# 	data = data₃
# 	zs = truezs₃
# 	μs, Σ, πs = LDA_fit(data, zs)
# 	mvnsₘₗ = [MvNormal(μ, Symmetric(Σ)) for μ in eachcol(μs)]
# 	ws, _ = e_step(data, mvnsₘₗ, πs)
# 	yŝ = argmax.(eachrow(ws))
# 	plt=plot(-6:.02:6, -6:0.02:6, (x,y) -> decisionBdry(x, y, mvnsₘₗ, πs)[2], c=1:3, alpha=0.6, title="LDA on a non-linear data", st=:heatmap, colorbar=false, ratio=1, framestyle=:zerolines)
# 	wrong_idx = findall(zs .!= yŝ)
# 	for k in [1,3,2]
# 		zks = findall(zs .== k)
# 		scatter!(data[zks, 1], data[zks, 2], c = k, alpha=0.6, label="")

# 	end
# 	acc = length(wrong_idx) / length(zs)

# 	if add_mis
# 		wrongzk = wrong_idx
# 		scatter!(data[wrongzk, 1], data[wrongzk, 2], c = :black, markersize=6, label="mis-classified",alpha=0.8, marker=:x, markerstrokewidth=2, title="LDA on a non-linear data; accuracy = "*L"%$(acc)")
# 	end

# 	plt
# end

# ╔═╡ 1b6cb7eb-814e-4df8-89ed-56ebc8f06a4a
md"""


## Model assumptions


It is not a big issue if we apply **QDA** on data with linear decision boundaries

* or when the co-variances are **shared** (the **LDA** assumption)
* assuming there are enough training data; otherwise it might overfit
"""

# ╔═╡ c4ab9540-3848-483c-9ba5-e795913f844a
# let
# 	class = 3
# 	gr()
# 	data = data₁
# 	# K = K₃
# 	zs = truezs₁
# 	μs, Σs, πs = QDA_fit(data, zs)
# 	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
# 	# mvns = []
# 	plt=plot(-5:.05:5, -5:0.05:5, (x,y) -> e_step([x, y]', mvns, πs)[1][class], levels=6, c=:coolwarm, lw=1, alpha=1.0, title="Decision boundary by supervised learning QDA", st=:contour, colorbar=false, ratio=1, framestyle=:grid)
# 	for k in [1,3,2]
# 		if k ==1
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		else
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		end
# 	end
# 	plt
# end

# ╔═╡ cba8b537-de68-4b2f-bf7a-d0bdd3aded7a
md"""

# MLE of QDA and LDA
"""

# ╔═╡ 4e038980-c531-4f7c-9c51-4e346eccc0aa
md"""

## MLE

In practice, we do not know the model parameters 

```math
\Large
\boldsymbol\Theta = \{\boldsymbol\pi, \{\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c\}_{c=1}^C\}
```



* the observed data ``\large \mathcal{D}=\{\mathbf{x}^{(i)}, y^{(i)}\}_{i=1}^n``



**Learning** or **training**: estimate a model's parameters given *observed* data

##### We use maximum likelihood estimation (MLE)

$$\Large\hat{\boldsymbol{{\Theta}}} = \arg\max_{\boldsymbol\Theta}\, \underbrace{\ln p(\mathcal{D}|\boldsymbol\Theta)}_{\mathcal{L}(\boldsymbol\Theta)}$$


## 

And we should simply solve it by calculus 

$\Large \frac{\partial \mathcal L}{\partial \boldsymbol\pi} =\mathbf 0; \frac{\partial \mathcal L}{\partial \boldsymbol\mu_k} =\mathbf 0; \frac{\partial \mathcal L}{\partial \boldsymbol\Sigma} =\mathbf 0$


"""

# ╔═╡ af373899-ed4d-4d57-a524-83b04063abf3
md"""

## MLE estimation of ``\boldsymbol\pi``

Recall that ``\large \boldsymbol\pi`` is the prior ``\large p(y)``'s parameter


$\large p(y^{(i)})= \mathcal{Cat}(y^{(i)}; \boldsymbol\pi)=\begin{cases}\pi_1 & y^{(i)}=1 \\ \pi_2 & y^{(i)} =2 \\\vdots & \vdots \\ \pi_C & y^{(i)} =C\end{cases}$

* ``y`` is also called multinoulli distributed
* and ``\pi_c``: the prior proportion of ``y^{(i)} =c``


"""

# ╔═╡ 6331b0f5-94be-426d-b055-e1369eb2a962
md"""

## MLE of Multinoulli ``\hat\pi``



"""

# ╔═╡ 06eebe92-bbab-449f-acb9-0e31ad2bfaa8
TwoColumnWideLeft(md"""

**Question**: In a few packages of M&Ms, I find in total 2620 M&Ms

| Red | Orange | Yellow | Green | Blue | Brown |
| --- | ------ | ------ | ------| -----| ----- |
| 372 | 544    | 369    | 483   | 481  |  371  |

> How to estimate the probability of each color ``\boldsymbol\pi``? 

""", html"<center><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Plain-M%26Ms-Pile.jpg/500px-Plain-M%26Ms-Pile.jpg' width = '200' /></center>")

# ╔═╡ 7c48e850-fdd9-4e77-87fa-a5800a26a77b
md"""

The **Maximum Likelihood Estimation (MLE)** for ``\boldsymbol{\pi}`` is just **relative frequency**


$$\large \hat{\pi}_c = \hat P(y= c) = \frac{n_c}{n} = \frac{\sum_{i=1}^n I(y^{(i)} = c)}{n}$$

* ``n`` is the total number of observations, that is 2620
* ``n_c`` is the number/count of ``c`` events, which is defined as 

  $n_c = \sum_{i=1}^n I(y^{(i)} = c)$

**For example,**

$$\hat{\pi}_{red} = \hat P(y = \text{red}) = \frac{372}{2620}$$
"""


# ╔═╡ a3ea595b-7b3e-4b97-bf1f-21f9a07fdd0d
md"""

## MLE estimation of single Gaussian



Given ``\{\mathbf{x}^{(i)}\}_{i=1}^n`` observations from a **single** Gaussian with unknown mean and variance 


```math
\mathbf{x}^{(i)} \sim \mathcal{N}(\boldsymbol\mu, \boldsymbol\Sigma)\;\;\; \text{for }i = 1\ldots n
```


The MLE estimators for ``\boldsymbol\mu, \boldsymbol\Sigma`` are


$$\large \hat{\boldsymbol{\mu}} = \frac{1}{n}{\sum_{i=1}^n \mathbf x^{(i)}}$$

$$\large \hat{\boldsymbol{\Sigma}} = \frac{1}{n} \sum_{i=1}^n (\mathbf x^{(i)}-\hat{\boldsymbol{\mu}})(\mathbf x^{(i)}-\hat{\boldsymbol{\mu}})^\top$$


* again very straighforward: sample mean and sample variance
"""

# ╔═╡ 8e4324d5-2a88-41d3-b229-43e9f41d4191
md"""

## 

Note that for uni-variate Gaussian, the MLE are just the sample average and variance


$$\large \hat{{\mu}} = \frac{1}{n}{\sum_{i=1}^n  x^{(i)}}$$

$$\large \hat{{\sigma}}^2 = \frac{1}{n} \sum_{i=1}^n ( x^{(i)}-\hat{{\mu}})( x^{(i)}-\hat{{\mu}}) = \frac{1}{n} \sum_{i=1}^n( x^{(i)}-\hat{{\mu}})^2$$


"""

# ╔═╡ 359b407f-6371-4e8a-b822-956173e89a47
md"""

## QDA learning: MLE 


To put them together, the **maximum likelihood estimators** are 

> $$\large \hat \pi_c = \frac{n_c}{n}$$
> $$\large \hat{\boldsymbol{\mu}}_c = \frac{1}{n_c}\, {\sum_{i=1}^n I(y^{(i)}=c)\cdot\mathbf x^{(i)}}$$
> $$\large \hat{\boldsymbol{\Sigma}_c} = \frac{1}{n_c} \sum_{i=1}^n I(y^{(i)}=c) (\mathbf x^{(i)}-\boldsymbol{\mu}_c)(\mathbf x^{(i)}-\boldsymbol{\mu}_c)^\top$$

* where  $n_c = \sum_{i=1}^n I(y^{(i)} = c)$
* ``\hat{\boldsymbol{\pi}}``: frequency of labels belong to each class 
* ``\hat{\boldsymbol{\mu}}_c, \hat{\boldsymbol{\Sigma}}_c``: the sample mean and covariance of the datasets belong to each class $c$
"""

# ╔═╡ 62f07c1e-4226-4a35-8d3a-198e41e10354
md"""

## Demonstration -- estimation of ``\pi``
"""

# ╔═╡ c1b120cb-36ec-49b9-af55-13e98630b6db
md"""

## Demonstration -- estimation of ``\mu``

For ``c=1\ldots C``:

$$\hat{\boldsymbol{\mu}}_c = \frac{1}{n_c}\, {\sum_{i=1}^n I(y^{(i)}=c)\cdot\mathbf x^{(i)}}$$
"""

# ╔═╡ ed986bfb-1582-4dcb-b39f-565657cfa59c
md"""

## Demonstration -- estimation of ``\mu, \Sigma``

For ``c=1\ldots C``:

$$\hat{\boldsymbol{\Sigma}_c} = \frac{1}{n_c} \sum_{i=1}^n I(y^{(i)}=c) (\mathbf x^{(i)}-\boldsymbol{\mu}_c)(\mathbf x^{(i)}-\boldsymbol{\mu}_c)^\top$$

"""

# ╔═╡ 92e04df6-153d-402d-a7fe-f708390c1185
md"""

## Another dataset


MLE are very efficient when the model assumptions are correct

* it means the estimates converge to the true parameters well
"""

# ╔═╡ af868b9b-130d-4d4f-8fc6-ff6d9b6f604f
md"True $\mu_1 = [-2 , 1];\mu_2 = [2 , 1]; \mu_3 = [0 , -1]$; 

Estimated ``\hat{\boldsymbol{\mu}}=``"

# ╔═╡ 08eb8c76-c19a-431f-b5ad-a14a38b18946
md"True $\Sigma_1 = \begin{bmatrix}.5, 0\\0, .5\end{bmatrix},  \Sigma_2 = \begin{bmatrix}.5, 0\\0, .5\end{bmatrix}, \Sigma_3 = \begin{bmatrix}0.5, 0\\0, 2\end{bmatrix}$;

Estimated ``\hat{\boldsymbol{\Sigma}}=``"

# ╔═╡ b0e16123-df7e-429c-a795-9e5ba788171a
πₘₗ = counts(truezs₂)/length(truezs₂);

# ╔═╡ bc04175a-f082-46be-a5ee-8d16562db340
md"True $\boldsymbol\pi = [0.2, 0.2, 0.6]$ ; 

Estimated ``\hat{\boldsymbol{\pi}}=`` $(latexify_md(πₘₗ))"

# ╔═╡ 05820b6f-45e9-4eaa-b6ba-c52813b5fe46
md"""

## LDA learning: MLE 


Then optimise $\mathcal L$: take derivatives and set to zero, we find the MLE



The **maximum likelihood estimators** are 

> $$\large \hat \pi_c = \frac{n_c}{n}$$
> $$\large\hat{\boldsymbol{\mu}}_c = \frac{1}{n_c}{\sum_{i=1}^n I(y^{(i)}=c)\cdot\mathbf x^{(i)}}$$
> $$\large\hat{\boldsymbol{\Sigma}} = \frac{1}{n}\sum_{c=1}^C \sum_{i=1}^n I(y^{(i)}=c) (\mathbf x^{(i)}-\boldsymbol{\mu}_c)(\mathbf x^{(i)}-\boldsymbol{\mu}_c)^\top$$


* ``\large \hat{\boldsymbol{\Sigma}}``: the pooled covariance of datasets 
"""

# ╔═╡ cdf72ed6-0d70-4901-9b8f-a12ceacd359d
md"""
## *Further details



Assume independence, the likelihood is

$P(\mathcal{D}|\Theta) = \prod_{i=1}^n p(y^i, x^i) = \prod_{i=1}^n p(y^i)p(x^i|y^i)$

Take log 

$$\mathcal L(\Theta) = \ln P(\mathcal{D}|\Theta) = \sum_{i=1}^n \ln p(y^i)+ \sum_{i=1}^n \ln p(x^i|y^i)$$




Write down the distribution with $I(\cdot)$ notation (you should verify they are the same)

$$p(y^i) = \prod_{k=1}^K \pi_k^{I(y^i=k)}$$ and also

$$p(x^i|y^i) = \prod_{k=1}^K \mathcal{N}(x^i|\mu_k,\Sigma_k)^{I(y^i=k)}$$

Their logs are 

$$\ln p(y^i) = \sum_{k=1}^K {I(y^i=k)} \cdot \pi_k\;\; \ln p(x^i|y^i) =  \sum_{k=1}^K {I(y^i=k)} \cdot \ln \mathcal{N}(x^i|\mu_k,\Sigma_k)$$

Then

$$\mathcal L(\Theta) = \sum_{i=1}^n \sum_{k=1}^K {I(y^i=k)} \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K {I(y^i=k)} \cdot \ln \mathcal{N}(x^i|\mu_k,\Sigma_k)$$

Therefore, we can isolate the terms and write $\mathcal L$ as a function of $\mu_k, \Sigma_k$:

$\mathcal L(\mu_k,\Sigma_k) = \sum_{i=1}^n {I(y^i=k)} \cdot \ln \mathcal{N}(x^i|\mu_k,\Sigma_k) +C$

which justifies why the MLE for $\mu_k, \Sigma_k$ are the pooled MLE for the k-th class's observations!


The first term is ordinary Multinoulli log-likelihood, its MLE is relative frequency (need to use Lagrange multiplier as $\sum_{k} \pi_k =1$).
"""

# ╔═╡ 2f8e92fc-3f3f-417f-9171-c2c755d5e0f0
begin
	μ_ml, Σ_ml = zeros(2,K₂), zeros(2,2,K₂)
	for k in 1:K₂
		data_in_ck = data₂[truezs₂ .==k,:]
		μ_ml[:,k] = mean(data_in_ck, dims=1)
		Σ_ml[:,:, k] = cov(data_in_ck)
	end
end

# ╔═╡ 5b980d00-f159-49cd-b959-479cd3b1a444
latexify((μ_ml[:, 1]); fmt=FancyNumberFormatter(2)),latexify((μ_ml[:, 2]); fmt=FancyNumberFormatter(2)), latexify((μ_ml[:, 3]); fmt=FancyNumberFormatter(2))

# ╔═╡ 58663741-fa05-4804-8734-8ccb1fa90b2d
Σsₘₗ = Σ_ml;

# ╔═╡ 928a1491-3695-4bed-b346-b983f389a26f
latexify((Σsₘₗ[:,:, 1]); fmt=FancyNumberFormatter(2)), latexify((Σsₘₗ[:,:, 2]); fmt=FancyNumberFormatter(2)), latexify((Σsₘₗ[:,:, 3]); fmt=FancyNumberFormatter(2))

# ╔═╡ 5d28e09c-891d-44c0-98a4-ef4cf3a235f1
μsₘₗ = μ_ml;

# ╔═╡ a0465ae8-c843-4fc0-abaf-0497ada26652
md"""

## Appendix

Utility functions
"""

# ╔═╡ dafd1a68-715b-4f06-a4f2-287c123761f8
begin
	function sampleMixGaussian(n, mvns, πs)
		d = size(mvns[1].Σ)[1]
		samples = zeros(n, d)
		# sample from the multinoulli distribution of cⁱ
		cs = rand(Categorical(πs), n)
		for i in 1:n
			samples[i,:] = rand(mvns[cs[i]])
		end
		return samples, cs
	end
end

# ╔═╡ e0cfcb9b-794b-4731-abf7-5435f67ced42
begin
	K₃ = 3
	trueπs₃ = [0.2, 0.6, 0.2]
	trueμs₃ = [[1.5, 1.5] [0.0, 0] [-1.5, -1.5]]
	trueΣs₃ = zeros(2,2,K₃)
	trueΣs₃ .= [2 -1.5; -1.5 2]
	trueΣs₃[:,:,2] = [2 1.5; 1.5 2]
	truemvns₃ = [MvNormal(trueμs₃[:,k], trueΣs₃[:,:,k]) for k in 1:K₃]
	n₃ = 200* K₃
	data₃, truezs₃ = sampleMixGaussian(n₃, truemvns₃, trueπs₃)
	data₃test, truezs₃test = sampleMixGaussian(100, truemvns₃, trueπs₃)
	xs₃ = (minimum(data₃[:,1])-1):0.1: (maximum(data₃[:,1])+1)
	ys₃ = (minimum(data₃[:,2])-1):0.1: (maximum(data₃[:,2])+1)
end;

# ╔═╡ d032f61d-c3fd-4e6c-92a3-9955e20f05b5
# using Flux

# ╔═╡ 1d08c5f5-cbff-40ef-bcb8-971637931e20
function LDA_fit(data, labels)
	n, d = size(data)
	sse = zeros(d, d)
	K = length(unique(labels))
	μs = zeros(d, K)
	ns = zeros(Int, K)
	for k in (unique(labels)|>sort)
		ns[k] = sum(labels .==k)
		datak = data[labels .== k, :]
		μs[:, k] = μk = mean(datak, dims=1)[:]
		error = (datak .- μk')
		sse += error' * error
	end
	μs, sse/n, ns/n
end

# ╔═╡ e4176cf0-d5b7-4b9a-a4cd-b25f0f5a987f
function QDA_fit(data, labels)
	n, d = size(data)
	# sse = zeros(d, d)
	K = length(unique(labels))
	μs = zeros(d, K)
	Σs = zeros(d,d,K)
	ns = zeros(Int, K)
	for k in (unique(labels)|>sort)
		ns[k] = sum(labels .==k)
		datak = data[labels .== k, :]
		μs[:, k] = μk = mean(datak, dims=1)[:]
		error = (datak .- μk')
		Σs[:,:,k] = error'*error/ns[k]
	end
	μs, Σs, ns/n
end

# ╔═╡ 30845782-cdd0-4c2e-b237-f331ee28db99
begin
	plt2_pen_qda = @df df scatter(:bill_length_mm, :bill_depth_mm, group =:species, framestyle=:origins,  alpha=0.2,xlabel="bill length (mm)", ylabel="bill depth (g)");
	df_ = df |> dropmissing
	ys_peng = df_[:, 1] 
	peng_map = Dict((s,i) for (i, s) in enumerate(unique(df_[:, 1])|>sort))
	ys_peng = [peng_map[s] for s in ys_peng]
	Xs_peng =  df_[:, 3:4]  |> Matrix
	pen_μs, pen_Σs, pen_πs = QDA_fit(Xs_peng, ys_peng)
	pen_mvns = [MvNormal(pen_μs[:,k], pen_Σs[:,:,k]) for k in 1:3]
	for k in 1:3
		plot!(32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=0.8, color=k, linewidth=3) 
		scatter!(pen_μs[1:1, k], pen_μs[2:2, k], c=k, label="", markershape=:x, markerstrokewidth=5, markersize=6)
	end
	plt2_pen_qda
end;

# ╔═╡ dd7db7a6-51c0-4421-8308-2c822107a370
TwoColumn(
html"<br/><br/><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gda_bnalone.svg
' width = '240' /></center>"

,  plot(plt2_pen_qda, size=(350,350)))

# ╔═╡ 2f7b3bf2-ce1a-4755-af3b-a82f02fb7752
let
	plt = @df df_ scatter(:bill_length_mm, :bill_depth_mm, group =:species,alpha=0.2,  framestyle=:origins,  xlabel="bill length (mm)", ylabel="bill depth (g)");
	# for k in 1:3
	k = 1
	nks = counts(ys_peng, 1:3)
		# plot!(32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=0.5, color=k, linewidth=3) 
	title = ""
	anim=@animate for k in 1:3
		title = title * " "*L"n_{%$k} = %$(nks[k]);"
		scatter!(Xs_peng[ys_peng .==k, 1], Xs_peng[ys_peng .==k, 2], c=k, label="", title=title)
	end
	# end
	# @animate for k in 1:K
	# 	scatter!()
	# end
	gif(anim, fps=1)
end

# ╔═╡ 9c7bbd1f-cf1c-4eae-a4c5-36324c5aff0a
nks = counts(ys_peng, 1:3);

# ╔═╡ de5879af-c979-4b3b-a444-db264c30297b
md"""The counts are $(latexify_md(nks)), therefore

```math
\Large
\hat{\boldsymbol\pi} = \begin{bmatrix}\frac{146}{333} \\ \frac{68}{333} \\ \frac{119}{333} \end{bmatrix}
```
"""

# ╔═╡ 6328dc99-9419-4ce0-9c76-ed2cadd8e2f3
let
	plt = @df df_ scatter(:bill_length_mm, :bill_depth_mm, group =:species,alpha=0.2,  framestyle=:origins,  xlabel="bill length (mm)", ylabel="bill depth (g)");
	# for k in 1:3
	k = 1
	nks = counts(ys_peng, 1:3)
		# plot!(32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=0.5, color=k, linewidth=3) 
	title = ""
	anim=@animate for k in 1:3
		title = title * " "*L"\hat{\mu}_{%$k} = %$(round.(pen_μs[:, k];digits=2));"
		scatter!(Xs_peng[ys_peng .==k, 1], Xs_peng[ys_peng .==k, 2], c=k, alpha=0.6, label="", title=title)

		scatter!(pen_μs[1:1, k], pen_μs[2:2, k], markershape=:x, markerstrokewidth=5, markersize=8,label="", c= k)
	end

	gif(anim, fps=0.8)
end

# ╔═╡ 170fc849-2f28-4d31-81db-39fbcc6ac6e4
let
	plt = @df df_ scatter(:bill_length_mm, :bill_depth_mm, group =:species,alpha=0.2,  framestyle=:origins,  xlabel="bill length (mm)", ylabel="bill depth (g)");
	# for k in 1:3
	k = 1
	nks = counts(ys_peng, 1:3)
		# plot!(32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=0.5, color=k, linewidth=3) 
	title = "Estimate"
	anim=@animate for k in 1:3
		title = title * " "*L"\{\mu_{%$k}, \Sigma_{%$(k)}\};"
		scatter!(Xs_peng[ys_peng .==k, 1], Xs_peng[ys_peng .==k, 2], c=k, alpha=0.6, label="", title=title)

		scatter!(pen_μs[1:1, k], pen_μs[2:2, k], markershape=:x, markerstrokewidth=5, markersize=8,label="", c= k)

		plot!(32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=0.5, color=k, linewidth=3) 
	end

	gif(anim, fps=0.8)
end

# ╔═╡ 93b4939f-3406-4e4f-9e31-cc25c23b0284
begin
	Xs = dropmissing(df)[:, 3:6] |> Matrix
	Ys = dropmissing(df)[:, 1]
	Ys_onehot  = Flux.onehotbatch(Ys, unique(Ys))
end;

# ╔═╡ 620789b7-59bc-4e17-bcfb-728a329eed0f
qdform(x, S) = dot(x, S, x)

# ╔═╡ 8d0c6fdc-4717-4203-b933-4b37fe60d512
# function logLikMixGuassian(x, mvns, πs, logLik=true) 
# 	l = logsumexp(log.(πs) .+ [logpdf(mvn, x) for mvn in mvns])
# 	logLik ? l : exp(l)
# end

# ╔═╡ d66e373d-8443-4810-9332-305d9781a21a
md"""

Functions used to plot and produce the gifs

"""

# ╔═╡ acfb80f0-f4d0-4870-b401-6e26c1c99e45
function plot_clusters(D, zs, K, loss=nothing, iter=nothing)
	title_string = ""
	if !isnothing(iter)
		title_string ="Iteration: "*string(iter)*";"
	end
	if !isnothing(loss)
		title_string *= " L = "*string(round(loss; digits=2))
	end
	plt = plot(title=title_string, ratio=1)
	for k in 1:K
		scatter!(D[zs .==k,1], D[zs .==k, 2], label="cluster "*string(k), ms=3, alpha=0.5)
	end
	return plt
end

# ╔═╡ e091ce93-9526-4c7f-9f14-7634419bfe57
# plot clustering results: scatter plot + Gaussian contours
function plot_clustering_rst(data, K, zs, mvns, πs= 1/K .* ones(K); title="")
	xs = (minimum(data[:,1])-0.5):0.1: (maximum(data[:,1])+0.5)
	ys = (minimum(data[:,2])-0.5):0.1: (maximum(data[:,2])+0.5)
	_, dim = size(data)
	# if center parameters are given rather than an array of MvNormals
	if mvns isa Matrix{Float64}
		mvns = [MvNormal(mvns[:,k], Matrix(1.0I, dim, dim)) for k in 1:K]
		πs = 1/K .* ones(K)
	end
	if ndims(zs) >1
		zs = [c[2] for c in findmax(zs, dims=2)[2]][:]
	end
	p = plot_clusters(data, zs, K)
	for k in 1:K 
		plot!(xs, ys, (x,y)-> pdf(mvns[k], [x,y]), levels=5,  st=:contour, colorbar = false, ratio=1, color=:jet, linewidth=2) 
		scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 5, markershape=:star4, markerstrokewidth=3, alpha=0.5,  framestyle=:origin)
	end
	title!(p, title)
	return p
end

# ╔═╡ 9d2e3f26-253e-4bba-b70f-fc3b5c4617d8
begin
	gr()
	plt2_= plot_clustering_rst(data₂,  K₂, truezs₂, truemvns₂, trueπs₂)
	title!(plt2_, "QDA example dataset")
	plt2_
end;

# ╔═╡ a92e040f-2e17-45e0-87e3-e29cca60c974
TwoColumnWideRight(md"""

\
\


$\boldsymbol\pi = [0.2, 0.2, 0.6]$

$\boldsymbol\mu_1 = [-2 , 0]; \boldsymbol\Sigma_1 = \begin{bmatrix}0.5, 0\\0, 0.5\end{bmatrix}$
$\boldsymbol\mu_2 = [2 , 0]; \boldsymbol\Sigma_2 = \begin{bmatrix}0.5, 0\\0, 0.5\end{bmatrix}$
$\boldsymbol\mu_3 = [0 , 0]; \boldsymbol\Sigma_3 = \begin{bmatrix}0.5, 0\\0, 2\end{bmatrix}$


""", plot(plt2_, size=(450,400)))

# ╔═╡ 0434dd27-4349-4731-80d5-b71ab99b53e2
begin
	gr()
	plt₃_= plot_clustering_rst(data₃,  K₃, truezs₃, truemvns₃, trueπs₃)
	title!(plt₃_, "More QDA example")
end;

# ╔═╡ 6be1c156-8dcc-48e3-b684-e89c4a0a7863
TwoColumn(md"""The true parameters are known:

$\boldsymbol\pi = [0.2, 0.6, 0.2]$

$\boldsymbol\mu_1 = [1.5 , 1.5];\,\boldsymbol \Sigma_1 = \begin{bmatrix}2, -1.5\\-1.5, 2\end{bmatrix}$
$\boldsymbol\mu_2 = [0.0 , 0.0];\, \boldsymbol\Sigma_2 = \begin{bmatrix}2, 1.5\\1.5, 2\end{bmatrix}$
$\boldsymbol\mu_3 = [-1.5 , -1.5];\,\boldsymbol \Sigma_3 = \begin{bmatrix}2, -1.5\\-1.5, 2\end{bmatrix}$


""", let
	# gr()
	# plt₃_= plot_clustering_rst(data₃,  K₃, truezs₃, truemvns₃, trueπs₃)
	# title!(plt₃_, "More QDA example")
	plot(plt₃_, size=(350,350))
end)

# ╔═╡ 889093e8-5e14-4211-8807-113adbac9a46
let
	gr()

	pltqda =  plot_clustering_rst(data₂,  K₂, truezs₂, truemvns₂, trueπs₂)
	# title!(plt2_, "QDA example dataset")
	# mvnsₘₗ = [MvNormal(μsₘₗ[:,k], Σsₘₗ[:,:,k]) for k in 1:K₂]

	μs, Σs, πs = QDA_fit(data₂, truezs₂)

	mvnsₘₗ = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	# pltqdaₘₗ = plot(title="QDA MLE params", ratio=1)
	# for k in 1:K₂
	# 	scatter!(data₂[truezs₂ .==k,1], data₂[truezs₂ .==k, 2], label="", c= k )
	# 	scatter!([mvnsₘₗ[k].μ[1]], [mvnsₘₗ[k].μ[2]], color = k, label = "", markersize = 10, markershape=:diamond, markerstrokewidth=3)
	# 	contour!(-5:.05:5, -5:0.05:5, (x,y)-> pdf(mvnsₘₗ[k], [x,y]), levels=5, colorbar = false, ratio=1,lw=3, c=:jet) 
	# end
	pltqdaₘₗ = plot_clustering_rst(data₂,  K₂, truezs₂, mvnsₘₗ, πs)
	title!(pltqda, "QDA Truth")
	title!(pltqdaₘₗ, "QDA MLE")
	plot(pltqda, pltqdaₘₗ, layout=(1,2))
end

# ╔═╡ d44526f4-3051-47ee-8b63-f5e694c2e609
function e_step(data, mvns, πs)
	K = length(mvns)
	# logLiks: a n by K matrix of P(dᵢ|μₖ, Σₖ)
	logLiks = hcat([logpdf(mvns[k], data') for k in 1:K]...)
	# broadcast log(P(zᵢ=k)) to each row 
	logPost = log.(πs') .+ logLiks
	# apply log∑exp to each row to find the log of the normalising constant of p(zᵢ|…)
	logsums = logsumexp(logPost, dims=2)
	# normalise in log space then transform back to find the responsibility matrix
	ws = exp.(logPost .- logsums)
	# return the responsibility matrix and the log-likelihood
	return ws, sum(logsums)
end

# ╔═╡ 901004da-2ac9-45fd-9643-8ce1cc819aa8
let
	Random.seed!(123)
	K₁ =3
	n₁ = 600
	# D₁ = zeros(n₁, 2)
	# 200 per cluster
		πs = 1/K₁ * ones(K₁)

	truezs₁ = repeat(1:K₁; inner=200)
	trueμs₁ = zeros(2, K₁)
	trueμs₁[:,1] = [2.0, 2.0]
	trueμs₁[:,2] = [-2.0, 2]
	trueμs₁[:,3] = [0., -1.5]
	LL = cholesky([1 0; 0. 1]).L
	data₁ = trueμs₁[:,1]' .+ randn(200, 2) * LL
	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(200, 2)* LL)
	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(200, 2)* LL)
	plt₁ = plot(ratio=1, framestyle=:origin)
	mvns = [MvNormal(trueμs₁[:, k], Matrix(I,2,2)) for k in 1:K₁]
	xs_qda = minimum(data₁[:,1])-0.1:0.1:maximum(data₁[:,1])+0.1
	ys_qda = minimum(data₁[:,2])-0.1:0.1:maximum(data₁[:,2])+0.1
	
	for k in 1:K₁
		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], c=k, label="Class"*string(k), alpha=0.8) 
		contour!(-5:.02:5, -5:0.02:5,  (x,y)-> pdf(mvns[k], [x,y]), levels=5, colorbar = false, lw=2, alpha=0.7, c=:jet) 
	end

	if add_db_1
		plot!(-5:.02:5, -5:0.02:5, (x,y) -> e_step([x, y]', mvns, πs)[1][:] |> argmax,  c=:black, lw=2, alpha=1.0, title="Decision boundary by LDA", st=:contour, colorbar=false, ratio=1)
	end
	plt₁
end

# ╔═╡ 3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
begin
	K₁ =3
	n₁ = 600
	# D₁ = zeros(n₁, 2)
	# 200 per cluster
	truezs₁ = repeat(1:K₁; inner=200)
	trueμs₁ = zeros(2, K₁)
	trueμs₁[:,1] = [0.0, 2.0]
	trueμs₁[:,2] = [2.0, 1]
	trueμs₁[:,3] = [0., -1.5]
	LL = cholesky([1 0.8; 0.8 1]).L
	Random.seed!(123)
	data₁ = trueμs₁[:,1]' .+ randn(200, 2)*LL'
	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(200, 2)* LL')
	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(200, 2)* LL')
	plt₁ = plot(ratio=1, framestyle=:origin)
	truemvns₁ = [MvNormal(trueμs₁[:, k], LL * LL') for k in 1:K₁]
	xs_qda = minimum(data₁[:,1])-0.2:0.05:maximum(data₁[:,1])+0.2
	ys_qda = minimum(data₁[:,2])-1:0.05:maximum(data₁[:,2])+0.5
	for k in 1:K₁
		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], c=k, ms=3, alpha=0.5, label="Class "*string(k)) 
		contour!(xs_qda, ys_qda, (x,y)-> pdf(truemvns₁[k], [x,y]), levels=5, colorbar = false, lw=2, alpha=0.9, c=:jet) 
	end
	title!(plt₁, "Classification (dataset with labels)")
	if add_db_2
		πs = 1/K₁ * ones(K₁)
		plot!(xs_qda, ys_qda, (x,y) -> e_step([x, y]', truemvns₁, πs)[1][:] |> argmax, c=:black, lw=2, alpha=0.7, title="Decision boundary by LDA", st=:contour, colorbar=false)
	end
	plt₁
end

# ╔═╡ 18a5208c-4a01-4784-a69d-0bd5e3bb9faf
md"""

## Summary 




| Model | Assumption | Example|
| :---|  :---:| :---:|
| LDA |  $\mathbf{\Sigma} =\mathbf\Sigma_1=\ldots \mathbf\Sigma_C$| $(plot(plt₁, size=(220,220), title="", legend=false))|
|  QDA  | $\mathbf{\Sigma}_1 \neq \mathbf\Sigma_2\neq\ldots \mathbf\Sigma_C$ |$(plot(plt₃_, size=(220,220), title="", legend=false))|
| Naive Bayes | Diagonal ``\mathbf{\Sigma}_c = \text{diag}(\boldsymbol{\sigma}_c)``|$(plot(plt2_, size=(220,220), title="", legend=false))|
"""

# ╔═╡ 629061d5-bf97-4ccf-af28-f1c5cd36b34c
let
	class = 3
	gr()
	data = data₂
	# K = K₃
	zs = truezs₂
	# μs, Σs, πs = QDA_fit(data, zs)
	μs, Σs, πs = trueμs₂, trueΣs₂, trueπs₂
	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	plt=plot(-5:.05:5, -5:0.05:5, (x,y) -> e_step([x, y]', mvns, πs)[1][class], levels=6, c=:coolwarm, lw=1, alpha=1.0, title="Decision boundary by QDA: "*L"p(y=%$(class)|\mathbf{x})", st=:contour, colorbar=true, ratio=1, framestyle=:origin)
	for k in [1,3,2]
		# if k ==1
			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, alpha=0.7, ms=3, label="class $(k)")
		# else
		# 	scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="class $(k)")
		# end
	end
	plt
end

# ╔═╡ 665c5bbc-9a5c-4e4a-930d-725bc2c9c883
let
	gr()
	data = data₃
	K = K₃
	zs = truezs₃
	# μs, Σs, πs = QDA_fit(data, zs)
	μs, Σs, πs = trueμs₃, trueΣs₃, trueπs₃
	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	# mvns = []
	plt=plot(-6:.05:6, -6:0.05:6, (x,y) -> e_step([x, y]', mvns, πs)[1][class_d3], levels=10, c=:coolwarm, lw=1, alpha=1.0, title="Pair-wise decision boundary by QDA", st=:contour, colorbar=true, ratio=1, framestyle=:origin)
	for k in [1,3,2]
		if k ==1
			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
		else
			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
		end
	end
	plt
end

# ╔═╡ d5ec14e2-fa45-4232-9ae8-06b84bf48525
let
	class = 3
	gr()
	data = data₁
	# K = K₃
	zs = truezs₁
	μs, Σs, πs = QDA_fit(data, zs)
	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	# mvns = []
	plt=plot(-7:.02:7, -6:0.02:6, (x,y) -> e_step([x, y]', mvns, πs)[1][:] |> argmax, c=1:3, lw=1, alpha=0.7, title="Decision boundary by QDA", st=:heatmap, colorbar=false, ratio=1, framestyle=:origins)
	for k in [1,3,2]
		if k ==1
			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
		else
			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
		end
	end
	plt
end

# ╔═╡ 7b47cda6-d772-468c-a8f3-75e3d77369d8
begin
# decision boundary function of input [x,y] 
function decisionBdry(x,y, mvns, πs)
	z, _ = e_step([x,y]', mvns, πs)
	findmax(z[:])
end

end

# ╔═╡ b4bfb1ba-73f2-45c9-9303-baee4345f8f6
let
	gr()
	zs= truezs₂
	data = data₂
	μs, Σs, πs = trueμs₂, trueΣs₂, trueπs₂
	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	ws, _ = e_step(data, mvns, πs)
	# yŝ = argmax.(eachrow(ws))
	# wrong_idx = findall(zs .!= yŝ)
	plt = plot(-5:.05:5, -5:0.05:5, (x,y) -> decisionBdry(x,y, mvns, πs)[2], c=1:3, alpha=0.5, ratio=1, framestyle=:origin, title="Decision boundary by QDA", st=:heatmap, colorbar=false)
	for k in [1,3,2]
		zks = findall(zs .== k)
		scatter!(data[zks, 1], data[zks, 2], c = k, alpha=0.3, label="")
	end
	# scatter!([data[iₜₕ, 1]], [data[iₜₕ, 2]], markersize = 12, markershape=:xcross, markerstrokewidth=3, c= :white, label=L"x^{(i)}")
	# pqda_class
	plt
end

# ╔═╡ eae40715-9b33-494f-8814-8c6f967aeade
plt_d3_true_bd=let
	gr()
	data = data₃
	K = K₃
	truezs = truezs₃

	# μs, Σs, πs = QDA_fit(data, truezs)
	μs, Σs, πs = trueμs₃, trueΣs₃, trueπs₃
	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	# mvns = []
	plt=plot(-7:.05:7, -7:0.05:7, (x,y) -> decisionBdry(x, y, mvns, πs)[2], c=1:3, alpha=0.6, title="Decision boundary by supervised learning QDA", st=:heatmap, colorbar=false, ratio=1, framestyle=:origin)
	for k in [1,3,2]
		if k ==1
			scatter!(data[truezs .==k, 1], data[truezs .==k, 2], c=k, label="")
		else
			scatter!(data[truezs .==k, 1], data[truezs .==k, 2], c=k, label="")
		end
	end
	plt
end;

# ╔═╡ 067d83e5-eb21-497c-9552-68dfde7cb085
plt_d3_true_bd

# ╔═╡ 3f869ae4-14b1-4b5a-b5c9-bc437bfc99da
TwoColumn(plot(plt_d3_true_bd, size=(330,330), title="QDA on non-linear data", framestyle=:origins, titlefontsize=10), let
	gr()
	data = data₃
	zs = truezs₃
	μs, Σ, πs = LDA_fit(data, zs)
	mvnsₘₗ = [MvNormal(μ, Symmetric(Σ)) for μ in eachcol(μs)]
	ws, _ = e_step(data, mvnsₘₗ, πs)
	yŝ = argmax.(eachrow(ws))
	plt=plot(-7:.02:7, -7:0.02:7, (x,y) -> decisionBdry(x, y, mvnsₘₗ, πs)[2], c=1:3, alpha=0.6, title="LDA on a non-linear data", st=:heatmap, colorbar=false, ratio=1, framestyle=:origins, size=(330,330),titlefontsize=10)
	wrong_idx = findall(zs .!= yŝ)
	for k in [1,3,2]
		zks = findall(zs .== k)
		scatter!(data[zks, 1], data[zks, 2], c = k, alpha=0.6, label="")

	end
	acc = length(wrong_idx) / length(zs)

	if add_mis
		wrongzk = wrong_idx
		scatter!(data[wrongzk, 1], data[wrongzk, 2], c = :black, markersize=6, label="mis-classified",alpha=0.8, marker=:x, markerstrokewidth=2, title="LDA on non-linear data; accuracy = "*L"%$(acc)", titlefontsize=10)
	end

	plt
end)

# ╔═╡ 27755688-f647-48e5-a939-bb0fa70c95d8
function m_step(data, ws)
	_, d = size(data)
	K = size(ws)[2]
	ns = sum(ws, dims=1)
	πs = ns ./ sum(ns)
	# weighted sums ∑ wᵢₖ xᵢ where wᵢₖ = P(zᵢ=k|\cdots)
	ss = data' * ws
	# the weighted ML for μₖ = ∑ wᵢₖ xᵢ/ ∑ wᵢₖ
	μs = ss ./ ns
	Σs = zeros(d, d, K)
	for k in 1:K
		error = (data .- μs[:,k]')
		# weighted sum of squared error
		# use Symmetric to remove floating number numerical error
		Σs[:,:,k] =  Symmetric((error' * (ws[:,k] .* error))/ns[k])
	end
	# this is optional: you can just return μs and Σs
	mvns = [MvNormal(μs[:,k], Σs[:,:,k]) for k in 1:K]
	return mvns, πs[:]
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
PalmerPenguins = "8b842266-38fa-440a-9b57-31493939ab85"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Clustering = "~0.15.3"
DataFrames = "~1.6.0"
Distributions = "~0.25.98"
Flux = "~0.14.0"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.21"
LogExpFunctions = "~0.3.24"
MLJLinearModels = "~0.9.2"
PalmerPenguins = "~0.1.4"
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
project_hash = "aefa0d645048c8289b88fb31f3fb258e03468832"

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

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

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
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "dd3000d954d483c1aad05fe1eb9e6a715c97013e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.22.0"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fe2838a593b5f776e1597e086dcd47560d94e816"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.3"

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
git-tree-sha1 = "6e8d74545d34528c30ccd3fa0f3c00f8ed49584c"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.11"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

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

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "f0af9b12329a637e8fba7d6543f915fff6ba0090"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.4.2"

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

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "e0a829d77e750a916a52df71b82fde7f6b336a92"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.14.1"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxMetalExt = "Metal"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

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

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "d73afa4a2bb9de56077242d98cf763074ab9a970"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.9"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f61f768bf090d97c532d24b64e07b237e9bb7b6b"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.9+0"

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
git-tree-sha1 = "cb56ccdd481c0dd7f975ad2b3b62d9eda088f7e2"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.14"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "83e95aaab9dc184a6dcd9c4c52aa0dc26cd14a1d"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.21"

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

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

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
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

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

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6698ab5e662b47ffc63a82b2f43c1cee015cf80d"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.0"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

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

[[deps.MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "Optim", "Parameters"]
git-tree-sha1 = "c92bf0ea37bf51e1ef0160069c572825819748b8"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.9.2"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c8b7e632d6754a5e36c0d94a4b466a5ba3a30128"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.8.0"

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

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "3d42748c725c3f088bcda47fa2aca89e74d59d22"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.4"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

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

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "16776280310aa5553c370b9c7b17f34aadaf3c8e"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.19"

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

[[deps.PalmerPenguins]]
deps = ["CSV", "DataDeps"]
git-tree-sha1 = "e7c581b0e29f7d35f47927d65d4965b413c10d90"
uuid = "8b842266-38fa-440a-9b57-31493939ab85"
version = "0.1.4"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "542b1bd03329c1d235110f96f1bb0eeffc48a87d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.6"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "364898e8f13f7eaaceec55fd3d08680498c0aa6e"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.4.2+3"

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

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

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

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

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

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

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

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2222b751598bd9f4885c9ce9cd23e83404baa8ce"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.3+1"

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
# ╟─120a282a-91c1-11ec-346f-25d56e50d38c
# ╟─5aa0adbe-7b7f-49da-9bab-0a78108912fd
# ╟─646dd3d8-6092-4435-aee9-01fa6a281bdc
# ╟─093c4c78-6179-4196-8d94-e548621df69b
# ╟─16497eaf-3593-45e0-8e6a-6783198663c3
# ╟─f7e989fd-d955-4323-bdef-57d9ffbe5a18
# ╟─e136edb5-3e98-4355-83e2-55761eb8b15c
# ╟─2acf2c33-bd3b-4369-857a-714d0d1fc600
# ╟─6775b1df-f9dd-423e-a6ef-5c9b345e5f0f
# ╟─76c05a93-698c-4ee1-9411-8a518c4e56b0
# ╟─21803313-fa45-48ea-bd3f-a30bc7696215
# ╟─3a747ab4-8dc1-48ef-97ae-a736448020b3
# ╟─501594f4-323f-4b99-93a4-8306e3228b63
# ╟─f7829734-0b56-43a4-ab26-37b74af5cfd8
# ╟─2496221f-70f9-4187-b329-35fbaf03a480
# ╟─9ecd708a-0585-4a68-b222-702e8de02abb
# ╟─133e4cb5-9faf-49b8-96ef-c9fc5c8e0b94
# ╟─c05d44bd-1564-430a-b633-b57cca1f5526
# ╟─fd0a8334-07e6-4678-bf28-d334d81fc67e
# ╟─2bed5f9e-3a01-4629-a92e-6ccca2d78119
# ╟─60c2d8e3-a411-4811-bcbd-2bbf52ff4ac3
# ╟─1a2c28b2-0821-4791-a52e-00b719cd1049
# ╟─4c10e534-8f0e-4767-9267-90d67a251dde
# ╟─cc9d9e19-c464-4ff7-ae1f-bba4111075b3
# ╟─200b1531-2a94-472a-9fd2-b90b04877581
# ╟─c4260ef4-d521-4d23-9495-41741940f128
# ╟─52b5327c-db4b-4f58-951a-ffba6677abd6
# ╟─8f61318f-6a39-45a6-a822-dc618b33c5cd
# ╟─1b9c9ccd-5d90-4e26-b6f4-da47c4c58619
# ╟─46f18f6b-04c6-47fd-b50c-9db3f90c8f4b
# ╟─bbfba1d5-c280-43e1-9721-d0ab2b9226ed
# ╟─9938af3d-0cad-4005-b2f6-db7fcf84b89a
# ╟─dd7db7a6-51c0-4421-8308-2c822107a370
# ╟─f033fbb8-55b3-40a7-a2e1-db8777daffc6
# ╟─30845782-cdd0-4c2e-b237-f331ee28db99
# ╟─eb7b07f3-6efe-4163-a7c5-33ff783c391b
# ╟─1ea85299-83c9-4c7e-8f90-7f37455b3db1
# ╟─cc9d8ae5-9ed3-407c-b63c-e8c4ea1cd472
# ╟─9a0ea270-10d8-44f5-98a1-f6324572548e
# ╟─17c07bab-5d5a-4480-8e60-94ffc4d891ef
# ╟─9e21c264-1175-479f-a0bd-51b21c67ce36
# ╟─e0473be1-1ee0-42fe-95a1-cdd6c948fb35
# ╟─0ca6a1e1-6f91-42fa-84b5-7c3b9170e56a
# ╟─77aaff69-13bb-4ffd-ad63-62993e13f873
# ╟─e5a23ba6-7859-4212-8854-86b238332eef
# ╟─22e1fbc9-f0bd-4159-b92f-11c412a660e6
# ╟─85eeec0d-f69f-4cf9-a718-df255a948433
# ╟─1ee13166-f4cb-46e1-83c0-5365ca707b9a
# ╟─6b27d9f7-e74c-404a-8248-c67cbbfe8342
# ╟─46d30b0c-0cf5-4369-a81f-6c234043a8ea
# ╟─abd46485-03c9-4077-8ace-34f9b897bb04
# ╟─dd915c43-b3cc-4608-87b0-852b2d533b15
# ╟─0bab5178-5cbd-467e-959d-1f09b496d2af
# ╟─b10c879d-98c8-4617-91de-fba1d9499ba4
# ╟─57c4e40c-6760-4c88-896d-7ea8faf134e0
# ╟─1a6cb45d-1b26-4efa-bd40-f7a8e3bbd770
# ╟─c85b688c-fc8d-4dfa-98bd-9e43dd0b79d5
# ╟─ee10a243-052f-4c0f-8f0d-e16ad6ceb611
# ╟─df0719cb-fc54-456d-aac7-48237a96cbdd
# ╟─b4d619a1-8741-4902-86f8-cd8e84c9d785
# ╟─2ad600a2-4e5d-4af6-a18c-caaa516a542d
# ╟─e243fe55-ee3e-47dc-9a7a-4319e0e86f8e
# ╟─c63369ed-58ed-4dd3-9292-c6c265ad52ba
# ╟─30cf6d78-c541-4d2d-b455-cb365d52b5cd
# ╟─e33b07c9-fb73-405d-8ee0-6e6e88e32bab
# ╟─c363c7d2-5749-49cb-8643-57a1b9dda8eb
# ╟─51b1572f-5965-41a8-b6d6-061c48f9af0c
# ╟─89bfebb4-ee0d-46c8-956a-6a5088599ae6
# ╟─eb14fb68-6950-4374-adef-35b583bf99fb
# ╟─2c3f2b50-0a95-4577-add8-8bf72580a44f
# ╟─ec66c0b1-f75c-41e5-91b9-b6a358cd9c3c
# ╟─e28dda3f-fbcd-47f0-8f99-4f3a2087905d
# ╟─901004da-2ac9-45fd-9643-8ce1cc819aa8
# ╟─ea4a783e-6a18-4d4e-b0a2-bf4fd8070c7a
# ╟─3b937208-c0bb-42e2-99a2-533113a0d4e0
# ╟─3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
# ╟─1747e403-c5d6-471a-9f6e-fcf7ee8063d1
# ╟─bcc88b1b-4e37-4d68-8af2-9b3923634dfd
# ╟─7c19c8ca-e30b-4214-b6f4-ac9eeb7d1a35
# ╟─853de250-143e-4add-b50d-2c73d1bc7910
# ╟─a92e040f-2e17-45e0-87e3-e29cca60c974
# ╟─cc60e789-02d1-4944-8ad5-718ede99669c
# ╟─9d2e3f26-253e-4bba-b70f-fc3b5c4617d8
# ╟─ca168509-f902-4f01-a976-c0c0959b73f3
# ╟─629061d5-bf97-4ccf-af28-f1c5cd36b34c
# ╟─c91d2a54-acf8-488b-b4c9-c1281a76e237
# ╟─b4bfb1ba-73f2-45c9-9303-baee4345f8f6
# ╟─e5e0f46c-c0c3-4347-bd8f-33e2cd033cc8
# ╟─6be1c156-8dcc-48e3-b684-e89c4a0a7863
# ╟─e0cfcb9b-794b-4731-abf7-5435f67ced42
# ╟─0434dd27-4349-4731-80d5-b71ab99b53e2
# ╟─84a4326b-2482-4b5f-9801-1a09d8d20f5b
# ╟─716ff72f-2299-472c-84b8-17315e8edc48
# ╟─665c5bbc-9a5c-4e4a-930d-725bc2c9c883
# ╟─067d83e5-eb21-497c-9552-68dfde7cb085
# ╟─eae40715-9b33-494f-8814-8c6f967aeade
# ╟─6094abc7-96bb-4c3f-9156-b5de5d7873f6
# ╟─d4acae8a-8f27-4a2a-93d5-63d6b6b6db20
# ╟─8ca2a156-e0f8-453f-a02c-daf23681bf89
# ╟─d29b4fcb-d738-46e1-8f7d-e2df5913a067
# ╟─18a5208c-4a01-4784-a69d-0bd5e3bb9faf
# ╟─552da9d4-5ab9-427d-9287-feb1eca57183
# ╟─55f2aacf-a342-4d4c-a1b7-60c5c29ab340
# ╟─3f869ae4-14b1-4b5a-b5c9-bc437bfc99da
# ╟─5a963c3d-46cd-4697-8991-5d5e1bb9a9e5
# ╟─1b6cb7eb-814e-4df8-89ed-56ebc8f06a4a
# ╟─d5ec14e2-fa45-4232-9ae8-06b84bf48525
# ╟─c4ab9540-3848-483c-9ba5-e795913f844a
# ╟─cba8b537-de68-4b2f-bf7a-d0bdd3aded7a
# ╟─4e038980-c531-4f7c-9c51-4e346eccc0aa
# ╟─af373899-ed4d-4d57-a524-83b04063abf3
# ╟─6331b0f5-94be-426d-b055-e1369eb2a962
# ╟─06eebe92-bbab-449f-acb9-0e31ad2bfaa8
# ╟─7c48e850-fdd9-4e77-87fa-a5800a26a77b
# ╟─a3ea595b-7b3e-4b97-bf1f-21f9a07fdd0d
# ╟─8e4324d5-2a88-41d3-b229-43e9f41d4191
# ╟─359b407f-6371-4e8a-b822-956173e89a47
# ╟─62f07c1e-4226-4a35-8d3a-198e41e10354
# ╟─2f7b3bf2-ce1a-4755-af3b-a82f02fb7752
# ╟─de5879af-c979-4b3b-a444-db264c30297b
# ╟─9c7bbd1f-cf1c-4eae-a4c5-36324c5aff0a
# ╟─c1b120cb-36ec-49b9-af55-13e98630b6db
# ╟─6328dc99-9419-4ce0-9c76-ed2cadd8e2f3
# ╟─ed986bfb-1582-4dcb-b39f-565657cfa59c
# ╟─170fc849-2f28-4d31-81db-39fbcc6ac6e4
# ╟─92e04df6-153d-402d-a7fe-f708390c1185
# ╟─889093e8-5e14-4211-8807-113adbac9a46
# ╟─af868b9b-130d-4d4f-8fc6-ff6d9b6f604f
# ╟─5b980d00-f159-49cd-b959-479cd3b1a444
# ╟─08eb8c76-c19a-431f-b5ad-a14a38b18946
# ╟─928a1491-3695-4bed-b346-b983f389a26f
# ╟─bc04175a-f082-46be-a5ee-8d16562db340
# ╟─b0e16123-df7e-429c-a795-9e5ba788171a
# ╟─58663741-fa05-4804-8734-8ccb1fa90b2d
# ╟─5d28e09c-891d-44c0-98a4-ef4cf3a235f1
# ╟─05820b6f-45e9-4eaa-b6ba-c52813b5fe46
# ╟─cdf72ed6-0d70-4901-9b8f-a12ceacd359d
# ╟─2f8e92fc-3f3f-417f-9171-c2c755d5e0f0
# ╟─a0465ae8-c843-4fc0-abaf-0497ada26652
# ╠═dafd1a68-715b-4f06-a4f2-287c123761f8
# ╠═d032f61d-c3fd-4e6c-92a3-9955e20f05b5
# ╠═1d08c5f5-cbff-40ef-bcb8-971637931e20
# ╠═e4176cf0-d5b7-4b9a-a4cd-b25f0f5a987f
# ╠═93b4939f-3406-4e4f-9e31-cc25c23b0284
# ╠═620789b7-59bc-4e17-bcfb-728a329eed0f
# ╠═7b47cda6-d772-468c-a8f3-75e3d77369d8
# ╠═8d0c6fdc-4717-4203-b933-4b37fe60d512
# ╟─d66e373d-8443-4810-9332-305d9781a21a
# ╠═acfb80f0-f4d0-4870-b401-6e26c1c99e45
# ╠═e091ce93-9526-4c7f-9f14-7634419bfe57
# ╠═d44526f4-3051-47ee-8b63-f5e694c2e609
# ╠═27755688-f647-48e5-a939-bb0fa70c95d8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
