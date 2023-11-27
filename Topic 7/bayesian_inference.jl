### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 6a1f3828-667a-11ed-2ea0-154032459b81
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
	using HypertextLiteral
	# using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
end

# ╔═╡ 7c49ffb7-dc5f-4cd5-ada5-44719e24b8a6
begin
	# using DataFrames, CSV
	using MLDatasets
	# using Images
end

# ╔═╡ b741192c-d6b8-435f-b553-0aa5ebed984d
using MCMCChains

# ╔═╡ 18a77849-f40d-4eb6-aa5d-07f7d4ec007d
# using Images

# ╔═╡ 0b01ac40-7054-4671-a370-483cce6b2231
TableOfContents()

# ╔═╡ 2ba672c4-967f-499b-9077-30ece46cfd83
ChooseDisplayMode()

# ╔═╡ 07d11cca-30a6-414c-bdd6-176de80e8b33
md"""

# CS5914 Machine Learning Algorithms


#### Bayesian inference
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ a8ec7847-1054-4a82-b1a4-d78f0fcf6cba
# md"""

# ## This note


# * Binary classification problem with logistic regression (single neuron)
# * The probabilistic model for logistic regression
#   * Assume ``p(y^{(i)}|\mathbf{x}^{(i)})`` is Bernoulli
#   * Goodness: maximum likelihood estimation
#   * Learning: gradient descent
# * Why we need regularisation

# * Bayesian inference 
# """

# ╔═╡ 4f27365a-5ae8-4639-ba22-6fe577487bed
md"""

## Today





* ##### MLE is very bad: overfitting


* ##### MAP (regularisation) is slightly better: but still equally *bad*


* ##### the solution:  (full) Bayesian inference


* ##### Approximate inference next time: MCMC

"""

# ╔═╡ c42cc08c-6628-4ec5-acb0-6bd52409e2f0
md"""
## Recap: binary classification



"""

# ╔═╡ a0474cb7-be98-449b-a5a7-ef8bac52fda6
md"""

## Recap: Logistic regression function

```math
\texttt{logistic}(h(\mathbf{x})) = σ(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^\top \mathbf{x}}}
```
"""

# ╔═╡ 06bfedf9-f4e2-4885-b466-842362e4d739
wv_ = [1, 1] * 1;

# ╔═╡ ee737d2c-d7d9-464b-8cd5-91b0b97f07d0
html"<center><img src='https://carpentries-incubator.github.io/ml4bio-workshop/assets/logit_nodes.png
' width = '400' /></center>";

# ╔═╡ 69b98f1e-92e8-4008-8d50-51e6c20f82e0
md"""

## Recap: probabilistic logistic regression



> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
> \\
> 1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
> ```


* where 

$\sigma(\mathbf{w}^\top \mathbf{x}^{(i)}) = \frac{1}{1+e^{-\mathbf{w}^\top \mathbf{x}^{(i)}}}$
* short-hand notation: $\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})$


##### The joint likelihood becomes 

```math
p(\mathbf{y}|\mathbf{X}, \mathbf{w}) = \prod_{i=1}^n p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) 
```

* ``y^{(i)}`` are conditionally independent given ``\mathbf{w}``
"""

# ╔═╡ 2fcba198-e1d0-4ba8-bc51-8a73d9cea0c5
begin
	function logistic_loss(w, X, y)
		σ = logistic.(X * w)
		# deal with boundary cases such as σ = 0 or 1, log(0) gracefully
		# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
		# rather you should use xlogy and xlog1py
		-sum(xlogy.(y, σ) + xlog1py.(1 .-y, -σ))
	end
end;

# ╔═╡ 9c844e0d-6396-46e6-8de2-4c7bb18254c2
function ∇logistic_loss(w, X, y)
	σ = logistic.(X * w)
	X' * (σ - y)
end;

# ╔═╡ 6d5c414d-1402-4c48-a9fb-51714dddec74
# using FiniteDifferences

# ╔═╡ 061ddb95-7197-4f48-9b02-ea2455637ab5
# begin
# 	w_test = rand(3)
# 	∇logistic_loss(w_test, D₂, targets_D₂) - FiniteDifferences.grad(central_fdm(5,1), (x) -> logistic_loss(x, D₂, targets_D₂), w_test)[1]
# end

# ╔═╡ 9e00e2b5-4c8f-41f6-b433-934edfc742ef
# losses, wws=let
# 	# a very bad starting point: completely wrong prediction function
# 	ww = [0, -5, -5]
# 	γ = 0.02
# 	iters = 2000
# 	losses = zeros(iters+1)
# 	wws = Matrix(undef, 3, iters+1)
# 	losses[1] = logistic_loss(ww, D₃, targets_D₃)
# 	wws[:, 1] = ww 
# 	for i in 1:iters
# 		gw = ∇logistic_loss(ww, D₃, targets_D₃)
# 		# Flux.Optimise.update!(opt, ww, -gt)
# 		ww = ww - γ * gw
# 		wws[:, i+1] = ww 
# 		losses[i+1] = logistic_loss(ww, D₃, targets_D₃)
# 	end
# 	losses, wws
# end

# ╔═╡ 4476bdd5-e16d-4130-a157-25c39d46cf71
# anim_logis=let

# 	gr()
# 	# xs_, ys_ = meshgrid(range(-5, 5, length=20), range(-5, 5, length=20))

# 	anim = @animate for t in [1:25...]
# 		plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], st=:scatter, label="class 1", c=2)
# 		plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
# 		w₀, w₁, w₂ = (wws[:, t])
# 		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contour, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="Iteration: "*L"%$(t);"*" loss: " *L"%$(round(losses[t]; digits=1))", ratio=1)
# 	end

# 	anim
# 	# ∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * 1
# 	# quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
# 	# w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
# 	# w_d₂
# 	# plot!(-6:1:6, (x) -> - w_d₃[1]/w_d₃[3] - w_d₃[2]/w_d₃[3] * x, lw=4, lc=:gray, label="Decision boundary: "*L"h(\mathbf{x}) =0", title="Least square classifier fails")
# end

# ╔═╡ 037ad762-08f1-4b7e-9582-5b1b094afeb4
# anim_logis2=let

# 	gr()

# 	anim = @animate for t in [1:25...]
# 		plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], ones(sum(targets_D₃ .== 1)), st=:scatter, label="class 1", c=2)
# 		plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], zeros(sum(targets_D₃ .== 0)), st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
# 		w₀, w₁, w₂ = (wws[:, t])
# 		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="Iteration: "*L"%$(t);"*" loss: " *L"%$(round(losses[t]; digits=1))", ratio=1)
# 	end

# 	anim
# 	# ∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * 1
# 	# quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
# 	# w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
# 	# w_d₂
# 	# plot!(-6:1:6, (x) -> - w_d₃[1]/w_d₃[3] - w_d₃[2]/w_d₃[3] * x, lw=4, lc=:gray, label="Decision boundary: "*L"h(\mathbf{x}) =0", title="Least square classifier fails")
# end

# ╔═╡ f7b465c7-951b-4abd-9252-5260c1e2c8b2
md"""

## Recap: MLE overfitting



$\Large \mathbf{w} \rightarrow \infty, \;\;\text{loss} \rightarrow 0$

"""

# ╔═╡ 55ffe10e-899d-456f-ba62-12d51a072843
md"""
## MLE -- overfitting

The prediction function for a new test data ``\mathbf{x}_{test}`` is 
* *i.e.* MLE **plug-in** prediction

```math
p(y_{test} =1 |\hat{\mathbf{w}}_{\text{ML}}, \mathbf{x}_{test}) =\sigma(\hat{\mathbf{w}}_{\text{ML}}^\top\mathbf{x}_{test})
```

"""

# ╔═╡ 2ad1858b-fcd3-4cc8-be70-97746480eacd
md"""


## Recap: regularisation (or MAP)



#### ``L_2`` regularisation

```math
	\hat{\mathbf{w}}_{\text{MAP}}\leftarrow \arg\min_{\mathbf{w}} \underbrace{L(\mathbf{w}) + \frac{\lambda }{2} \mathbf{w}^\top \mathbf{w}}_{\text{regularised loss}}
```


"""

# ╔═╡ 57777610-f6d9-4891-9ac7-e1ac23bf5d2e

md"""
## Recap: MAP ⇔ Regularisation



##### Introduce a prior ``p(\mathbf{w})``: then find the maximum a posteriori (*MAP*) estimator

$$\large p(\mathbf{w}|\mathbf{y}, \mathbf{X}) \propto \underbrace{p(\mathbf{w})}_{\text{prior}}\; \underbrace{p(\mathbf{y}|\mathbf{X}, \mathbf{w})}_{\text{likelihood}}$$

Then we have shown last time MAP ⇔ Regularisation

$$\large 
\begin{array}{c}\hat{\mathbf{w}}_{\text{MAP}} \leftarrow \arg\max_{\mathbf{w}}\,  \ln {p(\mathbf{y}|\mathbf{X}, \mathbf{w})} + \ln  {p(\mathbf{w})}\\
\Updownarrow \\

\hat{\mathbf{w}}_{\text{Ridge}}\leftarrow \arg\min_{\mathbf{w}} \underbrace{L(\mathbf{w}) + \frac{\lambda }{2} \mathbf{w}^\top \mathbf{w}}_{\text{regularised loss}}

\end{array}$$
"""

# ╔═╡ cf104a92-22e8-4995-96d1-869b1ea1c4a2
function ∇logistic_regularised(w, X, y; λ = 0.02)
	∇logistic_loss(w, X, y) + λ * w
end;

# ╔═╡ daeaf752-7f43-4afb-abdc-73a3ad7f2301
md"""

## Comparison

"""

# ╔═╡ e30b60a4-0625-4f68-83bd-481b623616c9
md"""

## However -- MAP is still very bad


"""

# ╔═╡ b2888f9b-c40a-4ee9-bdbd-898eb89df5be
md"""


MAP prediction is still a plug-in prediction 

```math
\begin{align}
p(y_{test} =1 |\hat{\mathbf{w}}_{\text{ML}}, \mathbf{x}_{test}) &=\sigma(\hat{\mathbf{w}}_{\text{ML}}^\top\mathbf{x}_{test}) \\

&vs\\

p(y_{test} =1 |\hat{\mathbf{w}}_{\text{MAP}}, \mathbf{x}_{test}) &=\sigma(\hat{\mathbf{w}}_{\text{MAP}}^\top\mathbf{x}_{test})

\end{align}
```


* ###### still black and white: just the street becomes wider
* ###### but lack of nuiance: 
  8 data on the "curbs" of the street are predicted the same
"""

# ╔═╡ 4840b4b2-f8ad-4a43-95c1-750d2cb39524
md"""

## Preview: full Bayesian prediction

##### Full Bayesian inference is just awesome

* provides a very nuanced, reasonable and **intelligent** prediction 

  * attention to the finest details

* all uncertainties well accounted for -- it implies good generalisation performance!
"""

# ╔═╡ 7e0ede71-879f-4bae-9efd-47fce755bde5
md"""

## How it is done? -- cough example


Recall the `cough` example, which can be represented as a Bayesian network

"""

# ╔═╡ afb802a4-6e4c-4006-85ca-00a226300165
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/cough_bn.png' width = '100' /></center>"

# ╔═╡ 8e2a5a42-5b95-4abd-b7c9-998eb7be977a
TwoColumn(md"""
The prior for the unknown ``h`` is

$$P(h) = \begin{cases} 0.89 & h=\texttt{healthy} \\
0.1 & h=\texttt{cold}\\ 
0.01 & h=\texttt{cancer}\end{cases}$$


""", md"""

The CPF for ``\texttt{Cough}`` is 

$$P(\texttt{cough}=\texttt{true}|h) = \begin{cases} 0.05 & h = \texttt{healthy} \\
0.85 & h = \texttt{cold} \\
0.9 & h = \texttt{cancer}
\end{cases}$$

""")

# ╔═╡ e0219c4c-9f1c-4cb7-bdb5-233294386cba
md"""

## Prediction -- full Bayesian inference


##### _Now_, let's *predict* he is going to *cough again*
* add a new unknown node ``\texttt{Cough}_{again}``
"""

# ╔═╡ 44c09451-ec0f-4ab1-9b86-44036c296809
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/cough_pred_bn.png' width = '250' /></center>"

# ╔═╡ 927fcaf8-348a-417c-9790-1cecaf24b4e4
md"""
##

##### *Bayesian* prediction: religiously apply ``sum+product`` rule

```math
\begin{align}
p(&\texttt{cough}_{\text{again}} |\texttt{cough}) \propto p(\texttt{cough}_{\text{again}} , \texttt{cough}) \\
&= \sum_{h} p(h, \texttt{cough}_{\text{again}} , \texttt{cough}) \tag{sum rule}\\
&= \sum_{h} p(\texttt{cough}_{\text{again}}|h) \underbrace{p(\texttt{cough}|h) p(h)}_{= p(h, \texttt{cough})\propto \; p(h|\texttt{cough})} \tag{factoring prop}\\
&= \sum_{\small h\in \{\texttt{cancer}, \texttt{healthy}, \texttt{cold}\}} p(\texttt{cough}_{\text{again}}|h) p(h|\texttt{cough})
\end{align}
```

Bayesian prediction is **democratic** and **balanced** **weighted average**

* each unknown hypothesis ``h\in \{\texttt{cancer}, \texttt{healthy}, \texttt{cold}\}`` is used to make a prediction for another ``\texttt{cough}_{\text{again}}``

* each prediction is weighted by its corresponding posterior ``p(h|\texttt{cough})``

##

In comparison, MLE and MAP plug predictions are very poor plugged-in

```math
p(\texttt{cough}_{\text{again}} | h = \hat{h}_{\text{MLE}}= \texttt{cancer})
```

and


```math
p(\texttt{cough}_{\text{again}} | h = \hat{h}_{\text{MAP}}= \texttt{cold})
```

* prediction by a single *point* hypothesis
"""

# ╔═╡ 756014db-42ca-4b35-9767-23a3ce3c871f
md"""

## 	Recap: Bayesian networks nodes

"""

# ╔═╡ a3ab0eca-070b-4e3d-99b2-478362df598b
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/notations.png' width = '200' /></center>"

# ╔═╡ 7717a633-822e-4347-bc39-17caa6cbcbd0
md"""

## Bayesian logistic regression in BN

"""

# ╔═╡ 03471888-cdf2-4e32-a06c-381e1b66e6d7
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/logistic_first.png' width = '220' /></center>"

# ╔═╡ 56009360-91c9-4f18-84b6-0afb886dc777
TwoColumn(md"""

The prior ``p(\mathbf{w})`` 

```math
p(\mathbf{w}) = \prod_{j=1}^m \mathcal{N}({0}, 1/\lambda)
```
* ``m`` variate (``m`` features) independent zero mean Gaussians with variance ``1/\lambda``

""", md"""

The CPF for ``y^{(i)}`` is
Bernoulli with a logistic transformed bias

```math
p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}) = \begin{cases} \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})& y^{(i)} = 1\\ 1- \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})& y^{(i)} = 0\end{cases} 
```

""")

# ╔═╡ fc448fa2-f764-4d0a-b50c-6283ecebee81
aside(tip(md"""``\lambda = \frac{1}{\sigma^2}`` is the precision of the Gaussian prior.""" ))

# ╔═╡ 1dd1b222-6e5e-47f1-9c5c-cfa2f847075f
md"""


## Bayesian logistic regression for prediction

Now what if we want to predict ``\mathbf{x}_{test} \in \mathbf{D}_{test}``?
* add an unknown node
* the hyperparameter ``\lambda`` (Gaussian prior's precision) is also added for completeness
"""

# ╔═╡ 340944a7-7a14-4a68-984e-d6924fc13cf1
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/logistic_pred.png' width = '220' /></center>"

# ╔═╡ a67aa14f-f4c0-4214-813e-fcc56aee852c
md"""

## Full Bayesian prediction


All machine learning has the same objective is: predict unseen data:

```math
p(y_{test} |\mathbf{x}_{test}, \mathbf{y}, \mathbf{X})
```

Do full Bayesian inference:

```math
\begin{align}
p(y_{test} |\mathbf{x}_{test}, \mathbf{y}, \mathbf{X}) &\propto p(y_{test}, \mathbf{x}_{test}, \mathbf{y}, \mathbf{X}) \\
&= \int_{\mathbf{w}}p(\mathbf{w}, y_{test}, \mathbf{x}_{test}, \mathbf{y},\mathbf{X})\,\rm{d}\mathbf{w} \;\;\; \text{should've been integration}\\
&= \int_{\mathbf{w}} p(y_{test}|\mathbf{x}_{test}, \mathbf{w}) \underbrace{p(\mathbf{y}|\mathbf{X}, \mathbf{w})p(\mathbf{w})}_{\propto p(\mathbf{w}|\mathbf{y}, \mathbf{X})}\,\rm{d}\mathbf{w}\\
&= \int_{\mathbf{w}} p(y_{test}|\mathbf{x}_{test}, \mathbf{w})  \underbrace{p(\mathbf{w}|\mathbf{y}, \mathbf{X})}_{\text{posterior!}}\,\rm{d}\mathbf{w}
\end{align}
```


#### Again it is a weighted average of multiple predictions

* **ensemble method** by default: democratic and balanced
* we do not know the true ``\mathbf{w}``: we may as well just let every ``\mathbf{w} \in \mathbb{R}^m`` to predict then take average
  * the average is weighted by ``p(\mathbf{w}|\mathbf{y}, \mathbf{X})``
"""

# ╔═╡ b710b9c1-f842-4c97-bda3-b4e6f324a9c9
md"""

## Markov Chain Monte Carlo estimation


For logistic regression, the weighted average (actually an integration) 


```math
\begin{align}
p(y_{test} |\mathbf{x}_{test}, \mathbf{y}, \mathbf{X}) 
= \int_{\mathbf{w}} p(y_{test}|\mathbf{x}_{test}, \mathbf{w})  \;\underbrace{p(\mathbf{w}|\mathbf{y}, \mathbf{X})}_{\text{posterior!}}\;\rm{d}\mathbf{w}
\end{align}
```

* the integration is intractable 


We can resort to Monte Carlo method, ``M`` samples are drawn from the posterior

```math
\mathbf{w}^{(m)} \sim p(\mathbf{w}|\mathbf{X}, \mathbf{y})
```

Then approximate the integration with Monte Carlo method

```math
\begin{align}
\hat{p}_{MC}(y_{test}=1 |\mathbf{x}_{test}, \mathbf{y}, \mathbf{X}) 
&= \frac{1}{M}\sum_{m=1}^M p(y_{test}=1|\mathbf{x}_{test}, \mathbf{w}^{(m)})  \\
&= \frac{1}{M}\sum_{m=1}^M \sigma((\mathbf{w}^{(m)})^\top\mathbf{x}_{test}) 
\end{align}

```
"""

# ╔═╡ af74c191-0176-4197-945e-174c0456bd66
begin
	# monte carlo prediction
	# ws: MCMC samples, assumed a mc × 3 vector
	# demonstration for 2-d case
	prediction(ws, x1, x2) = mean(logistic.(ws * [1.0, x1, x2]))
end;

# ╔═╡ f09f3b01-623d-4cd0-9d6f-939787aed445
md"""

## Demonstration


100 ``\mathbf{w}^{(m)}`` induced predictions (dash gray lines) are plotted below
* the final prediction is an average of all ``M`` ensemble's prediction
* a very natural curved prediction surface emerges by this average
"""

# ╔═╡ 66e8bc6e-61dd-4b5d-8130-b96f03d92bf8
md"""

# Appendix


Reading for Bayesian machine learning

* [Bayesian Regression and Classification by *Christopher M. Bishop* and *Michael E. Tipping* 2003](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-nato-bayes.pdf)
"""

# ╔═╡ b07efeea-75c9-4847-ba45-752ff9b34d53
md"""

### Data generation
"""

# ╔═╡ af30c343-64bd-45e6-be64-59f1f0413cc5
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

# ╔═╡ 7c1ba478-cd86-4a8c-b246-143f4dbfc009
plt_binary_2d = let
	gr()
	plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label=L"y^{(i)} = 1", xlabel=L"x_1", ylabel=L"x_2", title="Binary classification example", c=2, size=(400,300))
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label=L"y^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6])
end;

# ╔═╡ 5e5574dd-33fe-4d83-8088-5974989f7f0b
TwoColumn(md"""
\
\
\

**input** features: ``\mathbf{x}^{(i)} \in \mathbb{R}^m``
* *e.g.* the right is a two-dimensional, *i.e.* ``m=2``


**output** label: ``y^{(i)} \in \{0,1\}``


""", plt_binary_2d)

# ╔═╡ 1e6c5c71-1b42-4b76-a6a3-3e183365716d
begin
	D1 = [
	    7 4;
	    5 6;
	    8 6;
	    9.5 5;
	    9 7
	]

	# D1 = randn(5, 2) .+ 2
	
	D2 = [
	    2 3;
	    3 2;
	    3 6;
	    5.5 4.5;
	    5 3;
	]

	# D2 = randn(5,2) .- 2

	D = [D1; D2]
	D = [ones(10) D]
	targets = [ones(5); zeros(5)]
	AB = [1.5 8; 10 1.9]
end;

# ╔═╡ c5ec2d52-cf7f-403f-bd21-a68ab3b97a9e
wws_map = let
	λ = 0.02
	wwₗ₂ = randn(3) * 0.25
	γ = 0.02
	iters = 20000
	# optₗ₂ = Descent(ηₗ₂)
	# logLikGradl2(x) = Zygote.gradient((w) -> logPosteriorLogisticR(w, m0, V0, D, targets)[1], x)
	wwsₗ₂ = Matrix(undef, 3, iters)
	for i in 1:iters
		gt = ∇logistic_regularised(wwₗ₂, D, targets)
		wwₗ₂ = wwₗ₂ - γ * gt
		wwsₗ₂[:, i] = wwₗ₂
	end
	wwsₗ₂
end;

# ╔═╡ 216a2619-8eb3-4edc-988f-7af88eb121b3
wws_mle = let
	ww = randn(3) * 0.25
	γ = 0.02
	iters = 20000
	wws = Matrix(undef, 3, iters)
	for i in 1:iters
		gt = ∇logistic_loss(ww, D, targets)
		ww = ww - γ * gt
		wws[:, i] = ww
	end
	wws
end;

# ╔═╡ 28c04f67-699d-4665-90ef-ed402eed0b2c
begin
	using GLM
	glm_fit = glm(D, targets, Bernoulli(), LogitLink())
end

# ╔═╡ d05c93b1-d1fb-473c-b269-fce2bb48f8e0
let
	gr()
	p2 = plot(D[targets .== 1, 2], D[targets .== 1, 3], ones(5), st=:scatter, label="class 1", c=2)
	plot!(D[targets .== 0, 2], D[targets .== 0, 3], zeros(5), st=:scatter, label="class 2",  legend=:topleft, c=1)
	w₀, w₁, w₂ = ww = coef(glm_fit)[:]
	plot!(2:0.1:10, 2:0.1:10, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5,  xlabel=L"x_1", ylabel=L"x_2", title="Loss: " *L"%$(round(logistic_loss(ww, D, targets) ; digits=20))")


	p1 = plot(D[targets .== 1, 2], D[targets .== 1, 3], ones(5), st=:scatter, label="class 1", c=2)
	plot!(D[targets .== 0, 2], D[targets .== 0, 3], zeros(5), st=:scatter, label="class 2",  legend=:topleft, c=1)
	w₀, w₁, w₂ = ww0 = ww/20
	plot!(2:0.1:10, 2:0.1:10, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5,  xlabel=L"x_1", ylabel=L"x_2", title="Loss: " *L"%$(round(logistic_loss(ww0, D, targets) ; digits=20))")
	plot(p1, p2)
end

# ╔═╡ 774a9247-1db7-48d6-b1f9-63bcce3924cd
p_map_surface_1 = let
	plotly()
	wmle = coef(glm_fit)[:]
	w = wws_map[:,end]
	minx, maxx = minimum(D[:,2])-0.5, maximum(D[:,2])+0.5
	miny, maxy = minimum(D[:,3])-0.5, maximum(D[:,3])+0.5
	p1=Plots.surface(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> logistic(wmle' * [1, x, y]), legend=:best, title="MLE Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2", zlabel=L"p(y=1|\ldots)", colorbar=:false)
	p2=Plots.surface(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> logistic(w' * [1, x, y]), legend=:best, title="MAP Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2", zlabel=L"p(y=1|\ldots)", colorbar=:false)
	plot(p1, p2)
end;

# ╔═╡ cfc53ba8-dc4e-4929-9866-ea879f33d720
p_map_surface_1

# ╔═╡ 2cec4c51-1686-4166-ac67-c317075d79a4
begin
	n3_ = 30
	extraD = randn(n3_, 2)/2 .+ [2 -6]
	D₃ = [copy(D₂); [ones(n3_) extraD]]
	targets_D₃ = [targets_D₂; zeros(n3_)]
	targets_D₃_ = [targets_D₂; -ones(n3_)]
end

# ╔═╡ 60237e6f-a695-461c-b27a-c0e461d29227
md"""
### Extra functions

"""

# ╔═╡ bde57876-f675-4b1d-8cb6-33ed2790d31b
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

# ╔═╡ 56b87cec-2869-4588-b933-bfa40eedb745
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

# ╔═╡ 8b9248c4-7811-4d9a-bf29-068c3eb4beef
linear_reg(X, y; λ = 1) = (X' * X + λ *I) \ X' * y

# ╔═╡ ad8ff0d7-f8f9-4c37-a769-81d6e14b7cdf
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ 8da31795-a4ca-4177-9d78-0f09170d6710
begin
	function ∇σ(w, x) 
		wx = w' * x
		logistic(wx) * (1-logistic(wx)) * x
	end
end

# ╔═╡ fd35b73a-fdbc-4ae9-aa45-0eeeb91698c5
md"""

### MCMC - Bayesian inference on single neuron

A basic Metropolis Hasting sampler which samples the following posterior

```math
\mathbf{w}^{(m)} \sim p(\mathbf{w}|\mathbf{X}, \mathbf{y})
```
"""

# ╔═╡ f054c1ba-651c-450b-b2e8-2d0f0efb18a7
begin
	function MCMCLogisticR(X, y, dim; mc= 1000, burnin=0, thinning=10, m₀= zeros(dim), λ = 1/1e2, qV= nothing)
		if isnothing(qV)
			qV = Symmetric(inv(6/(π^2) * X' * X))
		end
		postLRFun(w) = -(logistic_loss(w, X, y) + 0.5 * λ * (w' * w))
		mcLR = MHRWMvN((w) -> postLRFun(w), dim; logP = true, Σ = qV, x0=m₀, mc=mc, burnin=burnin, thinning= thinning)
		return mcLR
		# return wt, Ht
	end


	# Metropolis Hastings with simple Gaussian proposal
	function MHRWMvN(pf, dim; logP = true, Σ = 10. *Matrix(I, dim, dim), x0=zeros(dim), mc=5000, burnin =0, thinning = 1)
		samples = zeros(dim, mc)
		C = cholesky(Σ)
		L = C.L
		pfx0 = pf(x0) 
		j = 1
		for i in 1:((mc+burnin)*thinning)
			xstar = x0 + L * randn(dim)
			pfxstar = pf(xstar)
			if logP
				Acp = pfxstar - pfx0 
				Acp = exp(Acp)
			else
				Acp = pfxstar / pfx0 
			end
			if rand() < Acp
				x0 = xstar
				pfx0 = pfxstar
			end
			if i > burnin && mod(i, thinning) ==0
				samples[:,j] = x0
				j += 1
			end
		end
		return samples
	end
	
end

# ╔═╡ 46f1fbb9-59f4-4521-b4f2-9f4584abe1bf
mcmcLR = MCMCLogisticR(D, targets, 3; mc=2000);

# ╔═╡ fc1a2ca5-ab39-44e6-8199-86f1656a0c03
p_bayes_pred=begin
	gr()
	ppf(x, y) = prediction(mcmcLR[:,1:1000]', x, y)
	contour(-0:0.1:10, 0:0.1:10, ppf, xlabel=L"x_1", ylabel=L"x_2", fill=true,  connectgaps=true, line_smoothing=0.85, title="Bayesian prediction", c=:jet, alpha=0.9)
end;

# ╔═╡ ad561e65-a77d-4801-b470-c82aa7c86097
p_bayes_pred

# ╔═╡ effd0f06-ddfb-478b-b04e-e0fb8b72cead
mcmcLR

# ╔═╡ 2910d53f-dacb-4868-8b80-11f9365b1597
plot(Chains(mcmcLR', [L"w_0" , L"w_1", L"w_2"]))

# ╔═╡ a36b3e58-f705-4bc4-b9a5-2b6e8dfd0f89
md"""

### Extra plots
"""

# ╔═╡ a452c99a-08b5-4334-8a3d-3ab1403e9e32
p_freq_1= let
	gr()
	freq_coef = coef(glm_fit)'
	# prediction(ws, x1, x2) = logistic(ws * [1.0, x1, x2])
	minx, maxx = minimum(D[:,2])-0.5, maximum(D[:,2])+0.5
	miny, maxy = minimum(D[:,3])-0.5, maximum(D[:,3])+1
	p_freq_1 = Plots.contour(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(freq_coef, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="MLE Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"x_2")
	Plots.plot!(p_freq_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=2, label="class 1")
	Plots.plot!(p_freq_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=1, label="class 2")
	Plots.plot!(p_freq_1, AB[1:2, 1], AB[1:2, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, annotate = [(AB[1,1], AB[1,2], text("A", :top,:red, 9)), (AB[2,1], AB[2,2], text("B", :top, :red, 9))], markerstrokewidth = 1, markercolor=:red)
	markers = [2, 5, 6, 9]
	for m in markers
		p0 = round(prediction(freq_coef, D[m, 2], D[m, 3]), digits=2)
		annotate!(D[m, 2], D[m,3], text(L"\hat{\sigma}="*"$(p0)", :bottom, 10))
	end

	for i in 1:2
		p0 = round(prediction(freq_coef, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1], AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :bottom, :red, 10))
	end
	p_freq_1
end;

# ╔═╡ 2440623f-04ee-4d13-8282-4b56768cb2c6
p_freq_2= let
	gr()
	freq_coef = w = wws_map[:,end]'
	# prediction(ws, x1, x2) = logistic(ws * [1.0, x1, x2])
	minx, maxx = minimum(D[:,2])-0.5, maximum(D[:,2])+0.5
	miny, maxy = minimum(D[:,3])-0.5, maximum(D[:,3])+1
	p_freq_1 = Plots.contour(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(freq_coef, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="MAP/Regularised Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"x_2")
	Plots.plot!(p_freq_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=2, label="class 1")
	Plots.plot!(p_freq_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=1, label="class 2")
	Plots.plot!(p_freq_1, AB[1:2, 1], AB[1:2, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, annotate = [(AB[1,1], AB[1,2], text("A", :top,:red, 9)), (AB[2,1], AB[2,2], text("B", :top, :red, 9))], markerstrokewidth = 1, markercolor=:red)
	markers = [2, 5, 6, 9]
	for m in markers
		p0 = round(prediction(freq_coef, D[m, 2], D[m, 3]), digits=2)
		annotate!(D[m, 2], D[m,3], text(L"\hat{\sigma}="*"$(p0)", :bottom, 10))
	end

	for i in 1:2
		p0 = round(prediction(freq_coef, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1], AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :bottom, :red, 10))
	end
	p_freq_1
end;

# ╔═╡ 096c720e-0c25-4008-bc05-80744006f168
TwoColumn(plot(p_freq_1, size=(350,350), titlefontsize=10), plot(p_freq_2, size=(350,350), titlefontsize=10))

# ╔═╡ 5067de18-678f-4b32-87a1-e1fa84c69942
chain_array = mcmcLR';

# ╔═╡ 26a378d5-735f-49ee-8dfb-33074a2dfa49
p_bayes_2 = let
	gr()
	freq_coef = w = wws_map[:,end]'
	as_ = collect(range(-10, 15, 6))
	f(x, b) = -(w[1]+b) /w[3] - w[2] / w[3] * x
	bs = f.(as_, +3)
	AB = [as_ bs]
	minx, maxx = minimum(D[:,2])-20, maximum(D[:,2])+20
	miny, maxy = minimum(D[:,3])-20, maximum(D[:,3])+20
	p_bayes_1 = Plots.contour(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(chain_array, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="Bayesian Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2")
	Plots.plot!(p_bayes_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=1, label="class 1")
	Plots.plot!(p_bayes_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=2, label="class 2")
	for i in 1:size(AB)[1]
		Plots.plot!(p_bayes_1, AB[:, 1], AB[:, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, markerstrokewidth = 4, markercolor=:red)
		
		p0 = round(prediction(chain_array, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1]-1.2, AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :right, :blue, 10))
	end

	as2_ = collect(range(0, 23, 7))
	bs2 = f.(as2_, -5)
	AB = [as2_ bs2]
	for i in 1:size(AB)[1]
		Plots.plot!(p_bayes_1, AB[:, 1], AB[:, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, markerstrokewidth = 4, markercolor=:red)
		p0 = round(prediction(chain_array, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1]+1.2, AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :left, :red, 10))
	end
	p_bayes_1
end;

# ╔═╡ dd4de5f6-2cea-479e-8f3d-38f86dd2d750
TwoColumn(plot(p_bayes_pred, size=(350,350), titlefontsize=12), plot(p_bayes_2, size=(350,350), titlefontsize=12, legendfontsize=5, framestyle=:semi))

# ╔═╡ eade1785-e759-4e23-ab97-50d94360687d
let

	p_bayes_pred = Plots.plot(D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 5, markercolor=2, label="class 1", legend=:topright, xlim=[-1, 11], ylim=[-1,11])
	Plots.plot!(p_bayes_pred, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 5,  markercolor=1, label="class 2")
	mean_pred = mean(chain_array, dims=1)[:]

	b, k =  -chain_array[1, 1:2] ./	chain_array[1, 3]
	plot!(-2:12, (x) -> k*x+b,  lw=0.1, lc=:gray, label=L"\sigma^{(m)}\sim p(\sigma|\mathcal{D})")
	for i in 2:2:200
		b, k =  -chain_array[i, 1:2] ./	chain_array[i, 3]
		plot!(-2:12, (x) -> k*x+b,  lw=0.2, ls=:dash, lc=:gray, label="", title="Bayesian ensemble predictions")
	end
	# b, k =  - mean_pred[1:2] ./	mean_pred[3]
	# plot!(-2:12, (x) -> k*x+b,  lw=3, lc=3, label=L"\texttt{mean}(\sigma^{(r)})")
	p_bayes_pred

end

# ╔═╡ 53b09b80-f10c-493c-8127-19a06bf248d2
p_bayes_1 = let
	p_bayes_1 = Plots.contour(2-6:0.1:9+7, 2-6:0.1:8+7, (x, y) -> prediction(chain_array, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="Bayesian Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"x_2")
	Plots.plot!(p_bayes_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=1, label="class 1")
	Plots.plot!(p_bayes_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=2, label="class 2")
	# Plots.plot!(p_bayes_1, AB[1:2, 1], AB[1:2, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, annotate = [(AB[1,1], AB[1,2], text("A", :top,:red, 9)), (AB[2,1], AB[2,2], text("B", :top, :red, 9))], markerstrokewidth = 1, markercolor=:red)
	# markers = [2, 5, 6, 9]
	# for m in markers
	# 	p0 = round(prediction(chain_array, D[m, 1], D[m, 2]), digits=2)
	# 	annotate!(D[m, 1], D[m,2], text(L"\hat{\sigma}="*"$(p0)", :bottom, 10))
	# end

	# for i in 1:2
	# 	p0 = round(prediction(chain_array, AB[i, 1], AB[i, 2]), digits=2)
	# 	annotate!(AB[i, 1], AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :bottom, :red, 10))
	# end
	p_bayes_1
end;

# ╔═╡ 23cd44a0-beb5-4d82-89e2-a7d77aaf8b63
p_freq_surface_1 = let
	gr()
	freq_coef = coef(glm_fit)'
	minx, maxx = minimum(D[:,2])-0.5, maximum(D[:,2])+0.5
	miny, maxy = minimum(D[:,3])-0.5, maximum(D[:,3])+0.5
	Plots.surface(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(freq_coef, x, y), legend=:best, title="MLE Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"p(y=1|\ldots)")
end;

# ╔═╡ c9302e0c-786a-498a-9a66-a6b160ebfe57
p_freq_surface_1

# ╔═╡ f7c1e631-a25f-409b-826e-3c272da9b56d
p_bayes_surface_1 = let
	Plots.surface(0:0.1:10, 0:0.1:10, (x, y) -> prediction(chain_array, x, y), legend=:best, title="Bayesian Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"p(y=1|\ldots)")
end;

# ╔═╡ 690ae8aa-f816-4b8c-8c2c-d09f5025fab6
p_freq_3= let
	gr()
	freq_coef = w = wws_map[:,end]'
	as_ = collect(range(-10, 15, 5))
	f(x, b) = -(w[1]+b) /w[3] - w[2] / w[3] * x

	bs = f.(as_, +3)
	AB = [as_ bs]
	minx, maxx = minimum(D[:,2])-20, maximum(D[:,2])+20
	miny, maxy = minimum(D[:,3])-20, maximum(D[:,3])+20
	p_freq_1 = Plots.contour(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(freq_coef, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="MAP/Regularised Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"x_2")
	Plots.plot!(p_freq_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=2, label="class 1")
	Plots.plot!(p_freq_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=1, label="class 2")


	for i in 1:size(AB)[1]
		Plots.plot!(p_freq_1, AB[:, 1], AB[:, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, markerstrokewidth = 4, markercolor=:red)
		
		p0 = round(prediction(freq_coef, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1]-1.2, AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :right, :blue, 10))
	end
	as2_ = collect(range(0, 23, 5))
	bs2 = f.(as2_, -5)
	AB = [as2_ bs2]
	for i in 1:size(AB)[1]
		Plots.plot!(p_freq_1, AB[:, 1], AB[:, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, markerstrokewidth = 4, markercolor=:red)
		
		p0 = round(prediction(freq_coef, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1]+1.2, AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :left, :red, 10))
	end
	p_freq_1
end;

# ╔═╡ e8e2a39c-bc18-42dd-86bf-684426ad6b35
p_freq_3

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.102"
GLM = "~1.9.0"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
MCMCChains = "~6.0.3"
MLDatasets = "~0.7.13"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.52"
StatsBase = "~0.34.2"
StatsPlots = "~0.15.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "f2e026ab4dd89f03d71d9e4d70109713c721034b"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "87e63dcb990029346b091b170252f3c416568afc"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.4.2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

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

[[deps.AtomsBase]]
deps = ["LinearAlgebra", "PeriodicTable", "Printf", "Requires", "StaticArrays", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "995c2b6b17840cd87b722ce9c6cdd72f47bab545"
uuid = "a963bdd2-2df7-4f54-a1ee-49d51e6be12a"
version = "0.3.5"

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

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BufferedStreams]]
git-tree-sha1 = "4ae47f9a4b1dc19897d3743ff13685925c5202ec"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.1"

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

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.Chemfiles]]
deps = ["AtomsBase", "Chemfiles_jll", "DocStringExtensions", "PeriodicTable", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "82fe5e341c793cb51149d993307da9543824b206"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.41"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3743181e30d87c23d9c8ebd493b77f43d8f1890"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.4+0"

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

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

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

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
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

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
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
git-tree-sha1 = "3d5873f811f582873bb9871fc9c451784d5dc8c7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.102"

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

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "273bd1cd30768a2fddfa3fd63bbc746ed7249e5f"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8e2d86e06ceb4580110d9e716be26658effc5bfd"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "da121cbdc95b065da07fbb93638367737969693f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.8+0"

[[deps.GZip]]
deps = ["Libdl", "Zlib_jll"]
git-tree-sha1 = "6388a2d8e409ce23de7d03a7c73d83c5753b3eb2"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.6.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

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
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "ec7df74b7b2022e8252a8bfd4ec23411491adc3b"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.0"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

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

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "fc5d1d3443a124fde6e92d0260cd9e064eba69f8"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.1"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

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
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

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
git-tree-sha1 = "c11d691a0dc8e90acfa4740d293ade57f68bfdbb"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.35"

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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "95220473901735a0f4df9d1ca5b171b568b2daa3"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.13.2"

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
git-tree-sha1 = "a9d2ce1d5007b1e8f6c5b89c5a31ff8bd146db5c"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.2.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "7ca6850ae880cc99b59b88517545f91a52020afa"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.25+0"

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
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

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

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "f9a11237204bc137617194d79d813069838fcf61"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.1"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "0d097476b6c381ab7906460ef1ef1638fbce1d91"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.2"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "ed1cf0a322d78cee07718bed5fd945e2218c35a1"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.6"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "8778ea7283a0bf0d7e507a0235adfff38071493b"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "6.0.3"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "3e6db72c2ab9cadfa3278ff388473a01fc0cfb9d"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MLDatasets]]
deps = ["CSV", "Chemfiles", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Printf", "Requires", "SparseArrays", "Statistics", "Tables"]
git-tree-sha1 = "10bc70e4c875f1b2ca65cef3ef9ebe705ef936b5"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.13"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "03ae109be87f460fe3c96b8a0dbbf9c7bf840bd5"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.9.2"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "3504cdb8c2bc05bde4d4b09a81b01df88fcbbba0"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.3"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "781916a2ebf2841467cda03b6f1af43e23839d85"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.9"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

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
git-tree-sha1 = "6e4e90c2e2ef091ef50b91af65fa4bb09c3d0728"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.6"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

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

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

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
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

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
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "bf6085e8bd7735e68c210c6e5d81f9a6fe192060"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.19"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.PeriodicTable]]
deps = ["Base64", "Test", "Unitful"]
git-tree-sha1 = "9a9731f346797126271405971dfdf4709947718b"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.1.4"

[[deps.Pickle]]
deps = ["BFloat16s", "DataStructures", "InternedStrings", "Serialization", "SparseArrays", "Strided", "StringEncodings", "ZipFile"]
git-tree-sha1 = "2e71d7dbcab8dc47306c0ed6ac6018fbc1a7070f"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.3"

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
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

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
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

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
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

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
git-tree-sha1 = "609c26951d80551620241c3d7090c71a73da75ab"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.6"

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

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

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

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0adf069a2a490c47273727e029371b31d44b72b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.5"
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
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

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

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "5cf6c4583533ee38639f73b880f35fc85f2941e0"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.3"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "a7a664c91104329c88222aa20264e1a05b6ad138"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.3"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a1f34829d5ac0ef499f6d84428bd6b4c71f02ead"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

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

[[deps.TupleTools]]
git-tree-sha1 = "155515ed4c4236db30049ac1495e2969cc06be9d"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.4.3"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

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
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulAtomic]]
deps = ["Unitful"]
git-tree-sha1 = "903be579194534af1c4b4778d1ace676ca042238"
uuid = "a7773ee8-282e-5fa2-be4e-bd808c38a91a"
version = "1.0.0"

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
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "04a51d15436a572301b5abbb9d099713327e9fc4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.4+0"

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
version = "1.52.0+1"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─6a1f3828-667a-11ed-2ea0-154032459b81
# ╟─7c49ffb7-dc5f-4cd5-ada5-44719e24b8a6
# ╟─18a77849-f40d-4eb6-aa5d-07f7d4ec007d
# ╟─0b01ac40-7054-4671-a370-483cce6b2231
# ╟─2ba672c4-967f-499b-9077-30ece46cfd83
# ╟─07d11cca-30a6-414c-bdd6-176de80e8b33
# ╟─a8ec7847-1054-4a82-b1a4-d78f0fcf6cba
# ╟─4f27365a-5ae8-4639-ba22-6fe577487bed
# ╟─c42cc08c-6628-4ec5-acb0-6bd52409e2f0
# ╟─5e5574dd-33fe-4d83-8088-5974989f7f0b
# ╟─7c1ba478-cd86-4a8c-b246-143f4dbfc009
# ╟─a0474cb7-be98-449b-a5a7-ef8bac52fda6
# ╟─06bfedf9-f4e2-4885-b466-842362e4d739
# ╟─56b87cec-2869-4588-b933-bfa40eedb745
# ╟─ee737d2c-d7d9-464b-8cd5-91b0b97f07d0
# ╟─69b98f1e-92e8-4008-8d50-51e6c20f82e0
# ╟─2fcba198-e1d0-4ba8-bc51-8a73d9cea0c5
# ╟─9c844e0d-6396-46e6-8de2-4c7bb18254c2
# ╟─6d5c414d-1402-4c48-a9fb-51714dddec74
# ╟─061ddb95-7197-4f48-9b02-ea2455637ab5
# ╟─9e00e2b5-4c8f-41f6-b433-934edfc742ef
# ╟─4476bdd5-e16d-4130-a157-25c39d46cf71
# ╟─037ad762-08f1-4b7e-9582-5b1b094afeb4
# ╟─f7b465c7-951b-4abd-9252-5260c1e2c8b2
# ╟─d05c93b1-d1fb-473c-b269-fce2bb48f8e0
# ╟─55ffe10e-899d-456f-ba62-12d51a072843
# ╟─c9302e0c-786a-498a-9a66-a6b160ebfe57
# ╟─2ad1858b-fcd3-4cc8-be70-97746480eacd
# ╟─57777610-f6d9-4891-9ac7-e1ac23bf5d2e
# ╟─cf104a92-22e8-4995-96d1-869b1ea1c4a2
# ╟─c5ec2d52-cf7f-403f-bd21-a68ab3b97a9e
# ╟─216a2619-8eb3-4edc-988f-7af88eb121b3
# ╟─774a9247-1db7-48d6-b1f9-63bcce3924cd
# ╟─daeaf752-7f43-4afb-abdc-73a3ad7f2301
# ╟─cfc53ba8-dc4e-4929-9866-ea879f33d720
# ╟─096c720e-0c25-4008-bc05-80744006f168
# ╟─e30b60a4-0625-4f68-83bd-481b623616c9
# ╟─b2888f9b-c40a-4ee9-bdbd-898eb89df5be
# ╟─e8e2a39c-bc18-42dd-86bf-684426ad6b35
# ╟─4840b4b2-f8ad-4a43-95c1-750d2cb39524
# ╟─dd4de5f6-2cea-479e-8f3d-38f86dd2d750
# ╟─26a378d5-735f-49ee-8dfb-33074a2dfa49
# ╟─7e0ede71-879f-4bae-9efd-47fce755bde5
# ╟─afb802a4-6e4c-4006-85ca-00a226300165
# ╟─8e2a5a42-5b95-4abd-b7c9-998eb7be977a
# ╟─e0219c4c-9f1c-4cb7-bdb5-233294386cba
# ╟─44c09451-ec0f-4ab1-9b86-44036c296809
# ╟─927fcaf8-348a-417c-9790-1cecaf24b4e4
# ╟─756014db-42ca-4b35-9767-23a3ce3c871f
# ╟─a3ab0eca-070b-4e3d-99b2-478362df598b
# ╟─7717a633-822e-4347-bc39-17caa6cbcbd0
# ╟─03471888-cdf2-4e32-a06c-381e1b66e6d7
# ╟─56009360-91c9-4f18-84b6-0afb886dc777
# ╟─fc448fa2-f764-4d0a-b50c-6283ecebee81
# ╟─1dd1b222-6e5e-47f1-9c5c-cfa2f847075f
# ╟─340944a7-7a14-4a68-984e-d6924fc13cf1
# ╟─a67aa14f-f4c0-4214-813e-fcc56aee852c
# ╟─b710b9c1-f842-4c97-bda3-b4e6f324a9c9
# ╟─af74c191-0176-4197-945e-174c0456bd66
# ╟─f09f3b01-623d-4cd0-9d6f-939787aed445
# ╟─eade1785-e759-4e23-ab97-50d94360687d
# ╟─ad561e65-a77d-4801-b470-c82aa7c86097
# ╟─fc1a2ca5-ab39-44e6-8199-86f1656a0c03
# ╟─66e8bc6e-61dd-4b5d-8130-b96f03d92bf8
# ╟─b07efeea-75c9-4847-ba45-752ff9b34d53
# ╠═af30c343-64bd-45e6-be64-59f1f0413cc5
# ╠═1e6c5c71-1b42-4b76-a6a3-3e183365716d
# ╠═2cec4c51-1686-4166-ac67-c317075d79a4
# ╟─60237e6f-a695-461c-b27a-c0e461d29227
# ╠═bde57876-f675-4b1d-8cb6-33ed2790d31b
# ╠═8b9248c4-7811-4d9a-bf29-068c3eb4beef
# ╠═ad8ff0d7-f8f9-4c37-a769-81d6e14b7cdf
# ╠═8da31795-a4ca-4177-9d78-0f09170d6710
# ╟─fd35b73a-fdbc-4ae9-aa45-0eeeb91698c5
# ╠═f054c1ba-651c-450b-b2e8-2d0f0efb18a7
# ╠═46f1fbb9-59f4-4521-b4f2-9f4584abe1bf
# ╟─effd0f06-ddfb-478b-b04e-e0fb8b72cead
# ╠═b741192c-d6b8-435f-b553-0aa5ebed984d
# ╟─2910d53f-dacb-4868-8b80-11f9365b1597
# ╟─a36b3e58-f705-4bc4-b9a5-2b6e8dfd0f89
# ╠═a452c99a-08b5-4334-8a3d-3ab1403e9e32
# ╠═2440623f-04ee-4d13-8282-4b56768cb2c6
# ╠═5067de18-678f-4b32-87a1-e1fa84c69942
# ╠═53b09b80-f10c-493c-8127-19a06bf248d2
# ╠═23cd44a0-beb5-4d82-89e2-a7d77aaf8b63
# ╠═f7c1e631-a25f-409b-826e-3c272da9b56d
# ╠═690ae8aa-f816-4b8c-8c2c-d09f5025fab6
# ╠═28c04f67-699d-4665-90ef-ed402eed0b2c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
