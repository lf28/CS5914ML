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
end

# ╔═╡ 4d6b2edb-3557-4aa8-b044-6096ece59cb8
using Flux

# ╔═╡ 92357262-d8ca-4da5-8949-5c1988e36c0e
begin
	using Logging
	Logging.disable_logging(Logging.Info); # or e.g. Logging.Info
end;

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ 52dcd4b9-9ef7-4128-a81d-d7e454cae9d6
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 19ebad08-8461-46fc-90bf-fcb1fa30d833
function show_img(path_to_file; center=true, h = 400, w = nothing)
	if center
		if isnothing(w)
			@htl """<center><img src= $(figure_url * path_to_file) height = '$(h)' /></center>"""
		else
			@htl """<center><img src= $(figure_url * path_to_file) width = '$(w)' /></center>"""
		end

	else
		if isnothing(w)
			@htl """<img src= $(figure_url * path_to_file) height = '$(h)' />"""
		else
			@htl """<img src= $(figure_url * path_to_file) width = '$(w)' />"""
		end
	end
end;

# ╔═╡ 7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
ChooseDisplayMode()

# ╔═╡ bc96a33d-9011-41ec-a19e-d472cbaafb70
md"""

# CS5914 Machine Learning Algorithms


#### Dealing with gradient problems
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
	topics = ["Gradients are very unstable: vanishing and explode", "Hack 1: activation function choice", "Hack 2: initialisation methods", "Hack 3: batch normalisation", "Hack 4: Residual connection"]
	@htl "<ul>$([@htl("""<li>$b</li><br>""") for b in topics[1:min(next_idx[1], length(topics))]])</ul>"
end

# ╔═╡ bc1ee08d-9376-44d7-968c-5e114b09a5e0
let
	next1
	next_idx[1] += 1
end;

# ╔═╡ aa89220b-91a1-4d0b-9626-4f3b2e688bce
md"""

## Deep neural network 


##### We have already showed *Deep* networks are better

* deep networks are more **flexible** than shallow networks
* their performance are "expected to beats" shallow networks


"""

# ╔═╡ 7b718599-7c4b-4685-b198-0fe47cc1770a
TwoColumn(md"""

$(show_img("CS5914/backprop/deepvsshallow.svg", w=350))
	
* "Number of regions": flexibility
* ``K``: number of layers
""", let
	delta = 0.26
	f_s(x) = log(x+delta)/ (x+delta) +0.05(x+delta) +2
	f_d(x) = log(10x) +.6* x
	plt1 = plot(0:0.05:10, f_s, label="Shallow networks", grid=true, xlim =(-0.2, 9), ylim =(-0.1,10),  yticks=true, xticks = true,ratio=0.59,  lc=:blue, xlabel="Amount of data", ylabel="Performance" , formatter=Returns(""), lw=2, framestyle=:semi, legend=false, titlefontsize=10, legendfontsize=8)
# plot(x, y, )
	plot!(0.01:0.1:10, f_d, label="Deep networks", lc=:red, lw=2, size=(350,240), title="Expected performance")
	x0 = 6

	annotate!([x0], [f_d(x0)+1.5], text("Deep networks", "Computer Modern",:red, 10))
	annotate!([x0], [f_s(x0)+0.5], text("Shallow networks", "Computer Modern", :blue, 10))

end)

# ╔═╡ 33749ea3-8b5c-4503-b194-fb33e10e0e1f
md"""

## Deep neural network 


##### We have already showed *Deep* networks are better


* deep networks are more **flexible** than shallow networks
* their performance "should be expected" to beat shallow networks (only if learned/trained properly)




##### But it also brings problems for learning !

* vanilla training methods for shallow network **does not** work
* ###### vanishing (and also exploding) problem
  * or in other words, numerical stability problem
  * makes the training difficult


"""

# ╔═╡ 34688842-5b3f-46d4-863d-bfafd794c885
show_img("CS5914/deep_problem.png", w=750)

# ╔═╡ 3b93d381-d38c-429d-9043-295348b46895
md"""

## Vanishing gradient problem -- example

##### An neural network with 8 hidden layer
* `Sigmoid` activation
* 8 neurons each layer


"""

# ╔═╡ 5f5cf4ee-1470-45c9-b9a9-fd5396fffc6a
md"""
## Vanishing gradient problem -- example

#### `Sigmoid` activation
"""

# ╔═╡ 5f27e9ad-faa5-4fb8-baa9-2a2b188af1f5
rand_u(u= 1.0f0) = (x...) -> Flux.rand32(x...) * u;

# ╔═╡ c3f8e1df-21ea-49a9-85b7-631eb0fd3752
md"""
`Tanh` does not work well either!
"""

# ╔═╡ d2e5a5fe-2b4e-4292-9eaa-71f5ecb7c8ea
md"""

## Why gradients _vanish_?
"""

# ╔═╡ 89060caa-803d-48be-9023-93897a6f4543
md"""

##### Forward pass with a ``K``--layer--network
"""

# ╔═╡ cb353447-3096-41fd-a278-1139a4b00fea
show_img("CS5914/backprop/multi_feedforward_net1.svg", w=750)

# ╔═╡ 78ba9723-ffb7-4974-82a7-11e5e9ba059b
md"""

## Why gradients _vanish_?
"""

# ╔═╡ 8c65e894-32cb-4abd-a86b-3a0f249f2a9e
md"""

##### Backward pass with a ``K``--layer--network


"""

# ╔═╡ c915f8a0-1ac8-45de-b551-3952867bf046
show_img("CS5914/backprop/multi_backward_net2.svg", w=750)

# ╔═╡ 1817095f-2d82-4fdb-928f-b8483525ec1d
md"""

**Backpropagation**: multiplies **all the local gradients**

```math
\begin{align}
\frac{\partial \ell}{\partial \mathbf{W}^{(k)}} &= \frac{\partial \ell}{\partial \mathbf{z}^{(K)}} \frac{\partial \mathbf{z}^{(K)}}{\partial \mathbf{a}^{(K-1)}} \frac{\partial \mathbf{a}^{(K-1)}}{\partial \mathbf{z}^{(K-1)}}  \frac{\partial \mathbf{z}^{(K-1)}}{\partial \mathbf{a}^{(K-2)}} \frac{\partial \mathbf{a}^{(K-2)}}{\partial \mathbf{z}^{(K-2)}}\ldots  \frac{\partial \mathbf{z}^{(k+1)}}{\partial \mathbf{a}^{(k)}} \frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}}\frac{\partial \mathbf{z}^{(k)}}{\partial \mathbf{W}^{(k)}}\\
\end{align}
```


* if any local gradients up there is zero (or even between (0, 1))
  * the gradients will **vanish** eventually
  * if ``0 \leq a  \leq 1``, ``a^{K} \rightarrow 0`` 

* if any local gradietns up there is too large (or $>1$)
  * the gradients will **explode** eventually (if infinite deep)
  * if ``|a| > 1``, ``a^{K} \rightarrow \infty`` 
"""

# ╔═╡ 878ada95-fb60-4b16-9980-e80403b25d83
md"""

## Why gradients _vanish_ ?



"""

# ╔═╡ 4e884b88-4324-4819-ab67-ef40f0b7a2d1
show_img("CS5914/backprop/multi_backward_net2.svg", w=750)

# ╔═╡ ecab42c1-b678-4636-a639-738f565a3b34
md"""

**Backpropagation**: multiplies **all the local gradients**

```math
\begin{align}
\frac{\partial \ell}{\partial \mathbf{W}^{(k)}} &= \frac{\partial \ell}{\partial \mathbf{z}^{(K)}} \boxed{\frac{\partial \mathbf{z}^{(K)}}{\partial \mathbf{a}^{(K-1)}} \frac{\partial \mathbf{a}^{(K-1)}}{\partial \mathbf{z}^{(K-1)}}}  \,\boxed{\frac{\partial \mathbf{z}^{(K-1)}}{\partial \mathbf{a}^{(K-2)}} \frac{\partial \mathbf{a}^{(K-2)}}{\partial \mathbf{z}^{(K-2)}}}\ldots  \boxed{\frac{\partial \mathbf{z}^{(k+1)}}{\partial \mathbf{a}^{(k)}} \color{red}{\frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}}}}\frac{\partial \mathbf{z}^{(k)}}{\partial \mathbf{W}^{(k)}}\\
\end{align}
```

###### Let's inspect some of the terms, *e.g.*

```math
{\color{red}{\frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}}} = \text{act}'(\mathbf{z}^{(k)}) }\quad \# \text{abuse of notation}
```


* ``\text{act}'(\mathbf{z}^{(k)})``: the gradient of the activation function
"""

# ╔═╡ 820664df-30f0-4290-be4a-12f9cae69404
aside(tip(md"""

To be more formal, the local gradient should be a diagonal matrix (known as Jacobian matrix):

```math
\small
\begin{align}
&\frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}} = \texttt{diag}(\text{act}'(\mathbf{z}^{(k)}))\\
&=\begin{bmatrix}\text{act}'(z^{(k)}_1) & & &&\\
& \text{act}'(z^{(k)}_2) &&&\\
&&&& \\
&&&\text{act}'(z^{(k)}_m)&
\end{bmatrix}
\end{align}
```


"""))

# ╔═╡ 0a10cdbb-92f0-45fa-be9e-ffdc144c27ea
md"""

## Why gradients _vanish_ ?


#### `Sigmoid` activation as an example


```math
\color{red}
\frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}} = \texttt{Sigmoid}'(\mathbf{z}^{(k)}) 
```


* ``\texttt{Sigmoid}'(z)``: the gradient of the activation function
"""

# ╔═╡ e20f00c1-352f-4e78-9ba2-91ac7251bdc0
md"Add saturate area: $(@bind add_sat CheckBox(default=false))"

# ╔═╡ 860f6d0e-5349-4193-bded-011023e546cb
TwoColumn(
	let
		x0 =6
	plt = plot(-10:0.1:10, sigmoid, lw=2, label=L"\texttt{Sigmoid}", title=L"\texttt{act}(z):\,\texttt{Sigmoid}",xlabel=L"z", size=(350,250), xlim =[-10,10])
		if add_sat
	vspan!([-11, -x0],  alpha=0.5, c=:gray, label="outputs saturate!")
	vspan!([x0, 11], alpha=0.5, c=:gray, label="", xlim =[-10,10])		
		end
	plt
	end, 
let
	
plt = plot(-10:0.1:10, (x) -> gradient(sigmoid, x)[1], lw=2,lc=2, label=L"\texttt{Sigmoid}'",title=L"\texttt{act}'(z):\, \texttt{Sigmoid}"*"'s grad",xlabel=L"z", size=(350,250), xlim =[-10,10])
x0 = 6 
	if add_sat
	vspan!([-11, -x0],  alpha=0.5, c=:gray, label="grad vanish!")
	vspan!([x0, 11], alpha=0.5, c=:gray, label="", xlim =[-10,10])		
		end
	plt
end)

# ╔═╡ 31328295-1059-4e78-8187-f4ec9ca3ad5f

md"""
# Hack 1: activation choice
"""

# ╔═╡ 943fcfba-b345-4de2-ab21-ed7df121cd4c
md"""

## Hack 1 -- do not use `Sigmoid`


##### Easy solution: try those *non-saturating activation functions*
* *e.g.* `ReLu`, `GeLu`, etc
"""

# ╔═╡ e072692a-009f-4300-bbdf-39dbe0ebaf50
TwoColumn(plot(-10:0.1:10, relu, lw=2, label=L"\texttt{ReLu}", title=L"\texttt{act}(z):\, \texttt{ReLu}",xlabel=L"z", size=(350,250)), 
	
	
	begin
	plot(-10:0.1:10, (x) -> gradient(relu, x)[1], lw=2,lc=1, label=L"\texttt{ReLu}'",title=L"\texttt{act}'(z):\, \texttt{ReLu}"*"'s grad",xlabel=L"z", size=(350,250))
plot!((x) -> gradient(sigmoid, x)[1], lw=2,lc=2, label=L"\texttt{Sigmoid}'",xlabel=L"z", size=(350,250))
	end
)

# ╔═╡ 1d807f6a-b8a4-43ef-9eb3-d84698b32b0f
md"""

# Hack 2: initialisation methods
"""

# ╔═╡ 2be60177-db90-4d9b-a16d-13e200e5a2f3
md"""

## But `ReLu` can still fail as well


 ###### Let's _increase_ ``K``:  # the layer (10 layers)

 ###### Also _increase_:  # of neurons per layer (100 neurons per layer)

* ###### _in other words_: _deeper_ also _wider_ network

###### _Initialise the weights_

```math
w \sim \mathcal{U}(0,1) \quad \quad \# \text{uniform between 0 and 1}
```

* random number between 0 and 1
"""

# ╔═╡ a0170ac9-be1f-4d2a-b96e-9ea296c4fb8a
md"""

## But `ReLu` can still fail as well


 ##### Let's _increase_ ``K``:  # the layer (10 layers)

 ##### Also _increase_:  # of neurons per layer (100 neurons per layer)

* ###### _in other words_: _deeper_ also _wider_ network
"""

# ╔═╡ e52bab81-8aac-4347-b6b4-193aeb1b0e40
md"""

Note the scale of ``y`` axis $10^{14}$: the gradients are very **un-stable**
* if continue training, it will **not** likely to converge!
"""

# ╔═╡ 746dc191-9126-49b2-8867-c75abd59cb62
rand_uniform(b = 1.0) = (x...) -> Flux.rand(x...) * (2*b) .- b;

# ╔═╡ 2b2a1dfb-6905-4850-a060-a9152e8c56b5
md"""

## Why? -- initialisation matters


"""

# ╔═╡ b2cfdac2-c4d5-4390-8d48-ee5283e6437d
TwoColumn(md"""

The **initialisation** used above is simple `np.random.rand()`

* *i.e.* random number between [0, 1]



For wider (``m`` neurons) and deeper network


```math
\small
z_1^{(l+1)} = b_1^{(l)} + \underbrace{w_{11}^{(l)}a^{(l)}_1 +  w_{12}^{(l)}a^{(l)}_2 +\ldots +w_{1m}^{(l)}a^{(l)}_m}_{m \text{ terms!}} 
```

""", 

show_img("CS3105/nnet-13.png", w=350)
)

# ╔═╡ 6b3312a3-bbf9-4ad0-b487-6c5c2756e649
md"""

## Why? -- initialisation matters


"""

# ╔═╡ 4f07d467-1dd3-47b4-8035-90cd92ed9a37
TwoColumn(md"""

The **initialisation** used above is simple `np.random.rand()`

* *i.e.* random number between

```math
w\sim \mathcal{U}(0, 1)
```



For wider (``m`` neurons) and deeper network


```math
\small
z_1^{(l+1)} = b_1^{(l)} + \underbrace{w_{11}^{(l)}a^{(l)}_1 +  w_{12}^{(l)}a^{(l)}_2 +\ldots +w_{1m}^{(l)}a^{(l)}_m}_{m \text{ terms!}} 
```

* if ``m`` increases, the sum ``z^{(l+1)}`` gets large

* compounding deep layers, the output explodes!
""", 

show_img("CS3105/nnet-13.png", w=350)
)

# ╔═╡ 55afd052-f6f9-4712-a3e5-5f63140a5d73
md"""
## Example with naive init




Increase the ``m`` (size of the input neurons) from 10 to 5000

```math
z_1= b_1 + \underbrace{w_{11}a_1 +  w_{12}a_2 +\ldots +w_{1m}a_m}_{m \text{ terms!}} 
```

* and plot the values for ``z``
* ``w``s are randomly intialised between -1 and 1

"""

# ╔═╡ 334523bb-2718-47ba-8b59-7c275cded63d
let
	Random.seed!(123)
	times = 200
	m = 10
	ws = rand(m, times) 
	ax = randn(m)

	# ws' * ax
	plt = plot(legend=:outerright, xlabel="# of neurons", ylabel="output: "*L"z")


	anim = @animate for m in 10:250:5000
		ax = randn(m)
		ws = rand(m, times) * 2 .- 1
		violin!(ws' * ax, label=L"m = %$(m)", xticks =[10:250:5000;])
	end

	gif(anim, fps=2)
end

# ╔═╡ 2177989b-734f-4ba0-8398-3e6e313ceee9
md"""

## Hack 2: better initialisations



##### `kaiming_uniform_`

\
Kaiming He (et al. 2015) introduces a simple method, for `ReLu`


```math
w \sim \mathcal{U}(-b, b),
```
where 
```math
b = \sqrt{2} \times \sqrt{\frac{3}{m}}
```

* ``m``: known as **fan-in**: the number of inputs from previous layer (or the column size of the weight matrix ``\mathbf{W}``)

* larger ``m``, smaller ``w``, more standardised hidden layer outputs
"""

# ╔═╡ dbadec7a-a312-4b69-b412-9bf1ddd7847c
md"""

## Hack 2: better initialisations


There are a wide range of other initialisations, *e.g.*

##### `xavier_uniform_`

\
Glorot, X. & Bengio, Y. (2010) introduces their method, for `Tanh`


```math
w \sim \mathcal{U}(-a, a)
```

* where 

```math
a = \frac{5}{3} \times \sqrt{\frac{6}{m_{\text{in}} + m_{\text{out}} }}
```

* ``m_{\text{in}}``: known as **fan-in**: the number of inputs from previous layer (or the number of column of the weight matrix ``\mathbf{W}``)
* ``m_{\text{out}}``: known as **fan-out**: the number of outputs from the layer (or the number of rows of the weight matrix ``\mathbf{W}``)
"""

# ╔═╡ 254d9fa5-e70b-435d-8f81-b91cee90942b
md"""
# Hack 3: Batch Normalisation
"""

# ╔═╡ a0fab847-cd21-4f02-aae2-68c91d67f6f0
md"""

## Towards hack 3: Batch Normalisation



> ##### The key of stable gradients is:
>
> * #### _stable outputs across the layers_



Clever initialisation does the trick for **initialisation** (and initialisation only)


* **however**, the weights changes during training 

* the standardisation effects **vane** over the time 

"""

# ╔═╡ 61ab1baa-e079-4538-a9b2-bdf0e30104a4
md"""

## Towards hack 3: Batch Normalisation



> ##### The key of stable gradients is:
>
> * #### _stable outputs across the layers_



Clever initialisation does the trick for **initialisation** (and initialisation only)


* **however**, the weights changes during training 

* the standardisation effects **vane** over the time 



> **Batch normalisation**: why not the standardise actively?

"""

# ╔═╡ a5bdfa24-b506-46bf-9e12-ba53a21e5779
md"""

## Recap: nomalisation trick for the input





"""

# ╔═╡ c6a4d653-9f9d-4d0c-857a-9c02e9c275a2
TwoColumn(md"""
\

Recall that we apply gradient descent with simple linear regression

* we usually standardise the input ``\mathbf{X}`` 
* such as each feature ``j`` have zero mean and unit standard deviation


""", md"""
$$\begin{align}
{\mu_j} = \frac{1}{N}\sum_{i=1}^N{x_j^{(i)}}
\\
\sigma_j = \sqrt{\frac{1}{N}\sum_{i=1}^N{(x_j^{(i)} - \mu_j)^2}}
\\
\text{scaled } x_j^{(i)}=\frac{x_j^{(i)}-\mu_j}{\sigma_j}\end{align}$$

""")

# ╔═╡ 5a3373e4-5643-482b-9f35-ebdb6fbc7559
show_img("CS5914/loss5.png", w=700)

# ╔═╡ 4d0490c4-b27e-423d-9298-35a60a3896c9
md"""

## Hack 3: batch normalisation



"""

# ╔═╡ 0f37f95c-3122-4e2f-8bfd-65309f9182f6
TwoColumn(md"""


> **Batch normalisation**: 
> standardise **each layers' output** such that: 
> * zero mean and unit standard devision
> * (optional) rescale at the end


* "batch": the mean and std are calculated with each mini-batch
""", 
md"""

For each data ``\mathbf{x}^{(i)} \in \mathcal{B}`` in batch ``\mathcal{B}``, each neuron ``j``

$$\small\begin{align}
{\mu_j} &= \frac{1}{N}\sum_{i=1}^N{x_j^{(i)}}
\\
\sigma_j &= \sqrt{\frac{1}{N}\sum_{i=1}^N{(x_j^{(i)} - \mu_j)^2}}
\\
x_j^{(i)}&=\frac{x_j^{(i)}-\mu_j}{\sqrt{\sigma_j^2 +\epsilon}}\\

\hat{x}_j^{(i)}&= \gamma x_j^{(i)} +\beta\tag{rescale}
\end{align}$$
"""
)

# ╔═╡ 934f9f71-a7a9-454a-b7f5-6419e1148227
html"""<center><img src="https://i0.wp.com/csmoon-ml.com/wp-content/uploads/2019/04/1WRio7MD4JDeLww-CyrxEbg.png?w=709" width = "500"/></center>""" 

# ╔═╡ ac5c1519-83c8-42fc-81a5-4523d19eef23
md"""

## Batch normalisation




##### Training phase

* The batch `mean`s and standard deviations (`std`s) are changing with each batch and also each iteration
* The algorithm therefore does not work with batch size = 1 (at least 20+ batch size)

* `model.train()` will set the module in training phase 
\


##### Testing phase

* We use the **global** `mean` and `std` across the batches in testing (or running statistics)
* check `track_running_stats` in PyTorch document for example
* `model.eval()` will set the module in testing phase 

"""

# ╔═╡ 54c23d2e-9c4e-4fd3-aa3d-67feaf1f632b
randn_σ(σ=1.0) = (x...) -> Flux.randn(x...) .* σ ;

# ╔═╡ 26de0c1c-db09-4f46-b23d-31ebbb6bf24b
md"""

## Example: batch normalisation

By using batch-normalisation


* the gradients are more stable
"""

# ╔═╡ e9803209-c3b7-4c21-9a35-86b69c0995ca
md"""
###### Without `BNnorm`

"""

# ╔═╡ 41a5afbb-c6ec-43b0-847d-9b7929f3d1b7
md"""


# Hack 4: Residual block 
"""

# ╔═╡ 4c08134f-9a0a-4a3a-90c7-fd9c9dfcbb38
md"""

## Why gradients _vanish_?
"""

# ╔═╡ bf9bda39-d081-4a68-8091-ae82f8ca8f35
md"""

##### Forward pass with a ``K``--layer--network
"""

# ╔═╡ 10be7f37-3526-48ca-b56f-d81a949f2fee
show_img("CS5914/backprop/multi_feedforward_net1.svg", w=750)

# ╔═╡ ee4830a9-4c0e-45b4-8fdc-74ec0e8d7fd6
md"""

##### Backward pass with a ``K``--layer--network
"""

# ╔═╡ 139c96a3-83da-4a5b-994b-f42bfaca83ab
show_img("CS5914/backprop/multi_backward_net2.svg", w=750)

# ╔═╡ 646c9815-bd63-4a31-ad04-e5d85df9efd0

md"""
Backpropagation: **multiplies** all the local gradients

```math
\begin{align}
\frac{\partial \ell}{\partial \mathbf{W}^{(k)}} &= \frac{\partial \ell}{\partial \mathbf{z}^{(K)}} \frac{\partial \mathbf{z}^{(K)}}{\partial \mathbf{a}^{(K-1)}} \frac{\partial \mathbf{a}^{(K-1)}}{\partial \mathbf{z}^{(K-1)}}  \frac{\partial \mathbf{z}^{(K-1)}}{\partial \mathbf{a}^{(K-2)}} \frac{\partial \mathbf{a}^{(K-2)}}{\partial \mathbf{z}^{(K-2)}}\ldots  \frac{\partial \mathbf{z}^{(k+1)}}{\partial \mathbf{a}^{(k)}} \frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}}\frac{\partial \mathbf{z}^{(k)}}{\partial \mathbf{W}^{(k)}}\\
\end{align}
```

> * **Multiplications** makes it unstable!
"""

# ╔═╡ 397269f9-ff8f-41cd-af41-5aefab6b2dba
md"""

## Skip connection


##### Pay attention to the `add` gate !
"""

# ╔═╡ 05d47e31-9ab0-4abb-bd77-019259b70437
show_img("CS5914/backprop/skip_connect.svg", w=750)

# ╔═╡ 47a405ca-963a-40d0-a116-45bf6cd848ca
md"""

## Recap: `add (+)` gate


`add (+)` gate

* it copies the same adjoint signal backward to both recipients
"""

# ╔═╡ 32c3e468-7353-4e4f-b1b0-65a1953d174d
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/plusgate.svg
# ' height = '300' /></center>"

show_img("CS5914/backprop/plusgate.svg", h=300)

# ╔═╡ 8dfc789e-8629-479e-86f9-32bfea9b4065
md"""

## Skip connection -- why it works


> ##### `add` gate effect
> * the **gradient** flows directly back
"""

# ╔═╡ d989fca3-a1a6-416a-928e-31ca6e166341
show_img("CS5914/backprop/skip_conn_back.svg", w=750)

# ╔═╡ 01a12e9e-9dcc-43bb-83e5-3ca2042c75e4
md"""

## Residual block -- put it together
"""

# ╔═╡ be267de9-2a9d-4b99-ae45-71cbcd53c41d
show_img("CS5914/backprop/residual.svg", w=750)

# ╔═╡ 0ed966e8-a242-4e96-b976-7174139cb279
md"""

##### or more commonly drawn like below
"""

# ╔═╡ 187aa1ec-fa10-4902-8bcd-72c86e391caf
html"""<center><img src="https://upload.wikimedia.org/wikipedia/commons/b/ba/ResBlock.png" width = "500"/></center>""" 

# ╔═╡ fc0b6698-c6a1-4315-8e5a-e7bc5234bd63
md"""
## `ResNet` (Kaiming He et al. 2015)
"""

# ╔═╡ 12f8c22c-1c6e-4bf1-9b46-cf25a12b91de
html"""<center><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/resnet_architecture.png" width = "700"/></center>""" 

# ╔═╡ 470f18e5-e0e1-411e-be46-df3cd5a43d6c
md"""

## The effect of Residual Block


ResNet makes it possible to train **deeper** network (*Kaiming et al. 2015*)
"""

# ╔═╡ c39a870b-fd23-4dab-8f53-306e3846c5d3
show_img("CS5914/backprop/resnet_results.png", w=750)

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

# ╔═╡ c363d75a-adc3-49fb-ba3c-bb4b870dd7db
begin
	# plotly()
	function gen_ball_data(n_obs=1000, n_dims=10; seed= 123)
		Random.seed!(seed)
		radius = sqrt(n_dims)
		points = randn(n_dims, n_obs)
		sphere = radius * points ./ sqrt.(sum(points.^2, dims=1))
	
		# Xs = sphere * 
		Xs = sphere .* rand(1, n_obs) .^ (1/ n_dims)
		adjustment = 1/ std(Xs)
		radius *= adjustment
		Xs *= adjustment
		ys = abs.(sum(Xs, dims=1)) .> radius * 0.5
		return Xs .|> Float32, ys .|> Float32
	end
end

# ╔═╡ b29505b8-6fda-4513-b1f2-59f78bd5d2c4
X, y= gen_ball_data(100)

# ╔═╡ 1d5093fc-9fd7-4976-8768-064e323c3f31
function build_nnet(n_input=2, n_output=1, n_hidden_units = 100, n_layers =5; act_fun= relu, use_bn=false, init_method = Flux.glorot_uniform)
	if n_hidden_units isa Vector
		n_units = n_hidden_units
		@assert length(n_units) == n_layers
	else
		n_units = repeat([n_hidden_units], n_layers)
	end
	layers = []
	if use_bn
		push!(layers, Dense(n_input, n_units[1], identity; init=init_method))
		push!(layers, BatchNorm(n_units[1], act_fun))
	else
		push!(layers, Dense(n_input, n_units[1], act_fun; init=init_method))
	end
	for l in 2:n_layers
		# push!(layers, Dense(n_units[l-1], n_units[l], act_fun; init=init_method))
		# if use_bn
		# 	push!(layers, BatchNorm(n_units[l]))
		# end

		if use_bn
			push!(layers, Dense(n_units[l-1], n_units[l], identity; init=init_method))
			push!(layers, BatchNorm(n_units[l], act_fun))
		else
			push!(layers, Dense(n_units[l-1], n_units[l], act_fun; init=init_method))
		end
	end

	push!(layers, Dense(n_units[end], n_output; init=init_method))

	return Chain(layers...)
end

# ╔═╡ c3324e87-5350-4663-afaf-90710be8721c
nnet_relu_fail = build_nnet(10, 1, 100, 10; act_fun = relu, use_bn=false, init_method = Flux.rand)

# ╔═╡ 640efe26-b14b-43fe-b926-f7e749461b9d
function produce_plots(hlist, glist; act_fun = sigmoid, init_method= "")
	plt1 = produce_plots(hlist; grad =false, act_fun = act_fun)
	plt2 = produce_plots(glist; grad =true, act_fun = act_fun)
	plot!(plt1, size=(350,350))
	plot!(plt2, size=(350,350))
	plot(plt1, plt2, size=(700,400), labelfontsize=12, titlefontsize=10)

end

# ╔═╡ 39b4fb9a-8776-47dd-8cb9-9b754e79d40e
function produce_plots(acts; grad =false, act_fun = :sigmoid)
	gr()
	title = "Hidden layer activations with " * L"\texttt{%$(act_fun)}"
	ylabel = "Hidden output"
	if grad
		title = "Gradients with " * L"\texttt{%$(act_fun)}"
		ylabel = "Gradients"
	end
	plt = violin(acts[1], label="Layer $(1)", legend=false, xlabel="Layers", ylabel=ylabel, title=title, titlefontsize =12)
	for (i, a) in enumerate(acts[2:end])
		violin!(a, label="Layer $(i+1)")
	end
	plt
end

# ╔═╡ 820a5289-ef20-4343-aeaf-2a8bb6cddcb2
function fb_pass(nnet; X=X, y = y, use_bn = false)
	acts_list =[]
	grad_list = []
	output = X


	for (li, layer) in enumerate(nnet[1:end-1])
		output = layer(output)
		push!(acts_list, vec((output)))
		# acts["a"*string(li)] = vec(output)
	end

	if use_bn
		acts_list = acts_list[2:2:end]
	end
	
	l, grad = Flux.withgradient(nnet) do m
		y_pred = m(X)
		Flux.Losses.logitbinarycrossentropy(y_pred, y)
	end
		
	for (li, gl) in enumerate(grad[1].layers[1:end-1])
		# output = layer(output)
		# # push!(acts, vec(copy(hl)))
		if :weight ∈ keys(gl) 
			push!(grad_list, vec((gl.weight)))
		end
		# acts["a"*string(li)] = vec(output)
	end
	acts_list, grad_list

	# acts_ = vcat(acts[1:end-1]...)
end

# ╔═╡ 5e86469a-23bd-46d1-ad96-8dbc26bc33ae
begin
	Random.seed!(4321)
	act_fun_ = sigmoid
	nnet_sigmoid = build_nnet(10, 1, 8, 8; act_fun = act_fun_, use_bn=false, init_method = rand_uniform())
	hlist, glist = fb_pass(nnet_sigmoid)
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	# plot!(plt, size=(700,420))
end;

# ╔═╡ 6bbfe970-16ee-4368-9778-b83ec12b71e6
nnet_sigmoid

# ╔═╡ 1369b978-bc1a-49ae-a97b-35a6b133dff8
TwoColumn(
plot!(produce_plots(hlist; act_fun=:Sigmoid), size=(350,350), titlefontsize=11)
,
plot!(produce_plots(glist; act_fun=:Sigmoid, grad=true), size=(350,350), ylim =(-0.015, 0.015),titlefontsize=11)
)

# ╔═╡ 69ab6f92-4818-4574-ab69-6fa655351b08
TwoColumn(
plot!(produce_plots(hlist; act_fun=:Sigmoid), size=(350,350), titlefontsize=11, title="Hidden layer activations with "*L"\texttt{Sigmoid}")
,
plot!(produce_plots(glist; act_fun=:Sigmoid, grad=true), title ="Vanishing gradient with "*L"\texttt{Sigmoid}", titlefontsize=11, size=(350,350), ylim =(-0.015, 0.015))
)

# ╔═╡ 65742069-c56d-461a-b53d-ab84f32045b5
let
	Random.seed!(4321)
	act_fun_ = tanh
	nnet_ = build_nnet(10, 1, 10, 8; act_fun = act_fun_, use_bn=false, init_method = rand_u(0.6))
	hlist_, glist = fb_pass(nnet_)
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	TwoColumn(
	plot!(produce_plots(hlist_; act_fun=act_fun_), size=(350,350), title="Hidden layer activations with "*L"\texttt{Tanh}")
	,
	plot!(produce_plots(glist; act_fun=act_fun_, grad=true), title="Gradients with "*L"\texttt{Tanh}", ylim = (-0.015, 0.015),size=(350,350))
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ 79c7fb43-ef44-4491-a3be-828b048f7088
let
	Random.seed!(123)
	act_fun_ = relu
	nnet_ = build_nnet(10, 1, 8, 8; act_fun = act_fun_, use_bn=false, init_method = rand_uniform())
	hlist, glist = fb_pass(nnet_)
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	TwoColumn(
	plot!(produce_plots(hlist; act_fun=act_fun_), size=(350,350), title="Hidden layer activations with "*L"\texttt{ReLu}")
	,
	plot!(produce_plots(glist; act_fun=act_fun_, grad=true), size=(350,350),title="Gradients with "*L"\texttt{ReLu}")
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ 50c24478-b20a-4c94-baa2-1762c3f64cce
begin
	Random.seed!(4321)
	# act_fun_ = relu
	# nnet_ = build_nnet(10, 1, 100, 10; act_fun = act_fun_, use_bn=false, init_method = Flux.rand)
	hlist_relu_fail, glist_relu_fail = fb_pass(nnet_relu_fail)

end;

# ╔═╡ e04abf99-5965-47d8-8bdb-88b655eec1ca
let
	act_fun_ = relu
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	TwoColumn(
	plot!(produce_plots(hlist_relu_fail; act_fun=act_fun_), size=(350,350), title="Activations: naive + "*L"\texttt{ReLu}")
	,
	plot!(produce_plots(glist_relu_fail; act_fun=act_fun_, grad=true), size=(350,350), title="Gradients: naive + "*L"\texttt{ReLu}" )
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ 37ff2521-aad5-4ffc-8dda-bb9c6bf4aa3e
let
	act_fun_ = relu
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	TwoColumn(
	plot!(produce_plots(hlist_relu_fail; act_fun=act_fun_), title="Activations: naive init + "*L"\texttt{ReLu}",size=(350,350), ytickfontsize=3)
	,
	plot!(produce_plots(glist_relu_fail; act_fun=act_fun_, grad=true),title="Gradients: naive init + "*L"\texttt{ReLu}", size=(350,350),ytickfontsize=3)
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ 4cc4fe34-656c-4b27-96f6-3eba0e82df84
let
	Random.seed!(4321)
	act_fun_ = relu
	nnet_ = build_nnet(10, 1, 100, 10; act_fun = act_fun_, use_bn=false, init_method = Flux.kaiming_uniform)
	hlist, glist = fb_pass(nnet_)
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	TwoColumn(
	plot!(produce_plots(hlist; act_fun=act_fun_), size=(350,350), title="Activations: kaiming + "*L"\texttt{ReLu}")
	,
	plot!(produce_plots(glist; act_fun=act_fun_, grad=true), title="Gradients: kaiming + "*L"\texttt{ReLu}", size=(350,350))
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ c6fa6270-d9e9-49f2-b167-1e1852a708b6
let
	Random.seed!(4321)
	act_fun_ = tanh
	nnet_ = build_nnet(10, 1, 100, 10; act_fun = act_fun_, use_bn=false, init_method = Flux.glorot_uniform)
	hlist, glist = fb_pass(nnet_)
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	TwoColumn(
	plot!(produce_plots(hlist; act_fun=act_fun_), size=(350,350), title="Activations: xavier + "*L"\texttt{Tanh}")
	,
	plot!(produce_plots(glist; act_fun=act_fun_, grad=true), title="Gradients: xavier + "*L"\texttt{Tanh}", size=(350,350))
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ cbbfa4b2-7de1-4ad5-9f75-af6468ae8415
let
	Random.seed!(4321)
	act_fun_ = relu
	nnet_ = build_nnet(10, 1, 100, 10; act_fun = act_fun_, use_bn=true, init_method = Flux.kaiming_uniform)
	hlist, glist = fb_pass(nnet_; use_bn=true)

	TwoColumn(
	plot!(produce_plots(hlist; act_fun=act_fun_), size=(350,350), title="Activations: kaiming + "*L"\texttt{ReLu} + \texttt{BNorm}", titlefontsize=11)
	,
	plot!(produce_plots(glist; act_fun=act_fun_, grad=true), title="Gradients: kaiming + "*L"\texttt{ReLu} + \texttt{BNorm}", ylim=(-0.4, 0.4) ,titlefontsize=11, size=(350,350))
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ 394910ab-987a-4678-9cc0-e2020511b9d2
let
	Random.seed!(4321)
	act_fun_ = relu
	nnet_ = build_nnet(10, 1, 100, 10; act_fun = act_fun_, use_bn=false, init_method = Flux.kaiming_uniform)
	hlist, glist = fb_pass(nnet_)
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	TwoColumn(
	plot!(produce_plots(hlist; act_fun=act_fun_), size=(350,350), title="Activations: kaiming + "*L"\texttt{ReLu}")
	,
	plot!(produce_plots(glist; act_fun=act_fun_, grad=true), ylim=(-0.4, 0.4), title="Gradients: kaiming + "*L"\texttt{ReLu}", size=(350,350))
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Flux = "~0.14.4"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.16.1"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.52"
StatsPlots = "~0.15.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "3194fe73d90a40a1107e497248ef181fecb679a1"

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

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

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
deps = ["Adapt", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "723a8ec75b26fe278256c89c363e370ba733c12e"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.14.4"

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

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "d73afa4a2bb9de56077242d98cf763074ab9a970"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.9"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1596bab77f4f073a14c62424283e7ebff3072eca"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.9+1"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

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
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

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
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

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
git-tree-sha1 = "af65afa916284e6c7e89f0ab974500cc9235618e"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "04a51d15436a572301b5abbb9d099713327e9fc4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.4+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

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
# ╟─4d6b2edb-3557-4aa8-b044-6096ece59cb8
# ╟─92357262-d8ca-4da5-8949-5c1988e36c0e
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─52dcd4b9-9ef7-4128-a81d-d7e454cae9d6
# ╟─19ebad08-8461-46fc-90bf-fcb1fa30d833
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─7091d2cf-9237-45b2-b609-f442cd1cdba5
# ╟─0a7f37e1-51bc-427d-a947-31a6be5b765e
# ╟─a696c014-2070-4041-ada3-da79f50c9140
# ╟─595a5ef3-4f54-4502-a943-ace4146efa31
# ╟─bc1ee08d-9376-44d7-968c-5e114b09a5e0
# ╟─aa89220b-91a1-4d0b-9626-4f3b2e688bce
# ╟─7b718599-7c4b-4685-b198-0fe47cc1770a
# ╟─33749ea3-8b5c-4503-b194-fb33e10e0e1f
# ╟─34688842-5b3f-46d4-863d-bfafd794c885
# ╟─3b93d381-d38c-429d-9043-295348b46895
# ╟─6bbfe970-16ee-4368-9778-b83ec12b71e6
# ╟─5e86469a-23bd-46d1-ad96-8dbc26bc33ae
# ╟─5f5cf4ee-1470-45c9-b9a9-fd5396fffc6a
# ╟─1369b978-bc1a-49ae-a97b-35a6b133dff8
# ╟─5f27e9ad-faa5-4fb8-baa9-2a2b188af1f5
# ╟─c3f8e1df-21ea-49a9-85b7-631eb0fd3752
# ╟─65742069-c56d-461a-b53d-ab84f32045b5
# ╟─d2e5a5fe-2b4e-4292-9eaa-71f5ecb7c8ea
# ╟─89060caa-803d-48be-9023-93897a6f4543
# ╟─cb353447-3096-41fd-a278-1139a4b00fea
# ╟─78ba9723-ffb7-4974-82a7-11e5e9ba059b
# ╟─8c65e894-32cb-4abd-a86b-3a0f249f2a9e
# ╟─c915f8a0-1ac8-45de-b551-3952867bf046
# ╟─1817095f-2d82-4fdb-928f-b8483525ec1d
# ╟─878ada95-fb60-4b16-9980-e80403b25d83
# ╟─4e884b88-4324-4819-ab67-ef40f0b7a2d1
# ╟─ecab42c1-b678-4636-a639-738f565a3b34
# ╟─820664df-30f0-4290-be4a-12f9cae69404
# ╟─0a10cdbb-92f0-45fa-be9e-ffdc144c27ea
# ╟─e20f00c1-352f-4e78-9ba2-91ac7251bdc0
# ╟─860f6d0e-5349-4193-bded-011023e546cb
# ╟─69ab6f92-4818-4574-ab69-6fa655351b08
# ╟─31328295-1059-4e78-8187-f4ec9ca3ad5f
# ╟─943fcfba-b345-4de2-ab21-ed7df121cd4c
# ╟─e072692a-009f-4300-bbdf-39dbe0ebaf50
# ╟─79c7fb43-ef44-4491-a3be-828b048f7088
# ╟─1d807f6a-b8a4-43ef-9eb3-d84698b32b0f
# ╟─2be60177-db90-4d9b-a16d-13e200e5a2f3
# ╟─c3324e87-5350-4663-afaf-90710be8721c
# ╟─a0170ac9-be1f-4d2a-b96e-9ea296c4fb8a
# ╟─e52bab81-8aac-4347-b6b4-193aeb1b0e40
# ╟─50c24478-b20a-4c94-baa2-1762c3f64cce
# ╟─e04abf99-5965-47d8-8bdb-88b655eec1ca
# ╟─746dc191-9126-49b2-8867-c75abd59cb62
# ╟─2b2a1dfb-6905-4850-a060-a9152e8c56b5
# ╟─b2cfdac2-c4d5-4390-8d48-ee5283e6437d
# ╟─6b3312a3-bbf9-4ad0-b487-6c5c2756e649
# ╟─4f07d467-1dd3-47b4-8035-90cd92ed9a37
# ╟─55afd052-f6f9-4712-a3e5-5f63140a5d73
# ╟─334523bb-2718-47ba-8b59-7c275cded63d
# ╟─2177989b-734f-4ba0-8398-3e6e313ceee9
# ╟─4cc4fe34-656c-4b27-96f6-3eba0e82df84
# ╟─37ff2521-aad5-4ffc-8dda-bb9c6bf4aa3e
# ╟─dbadec7a-a312-4b69-b412-9bf1ddd7847c
# ╟─c6fa6270-d9e9-49f2-b167-1e1852a708b6
# ╟─254d9fa5-e70b-435d-8f81-b91cee90942b
# ╟─a0fab847-cd21-4f02-aae2-68c91d67f6f0
# ╟─61ab1baa-e079-4538-a9b2-bdf0e30104a4
# ╟─a5bdfa24-b506-46bf-9e12-ba53a21e5779
# ╟─c6a4d653-9f9d-4d0c-857a-9c02e9c275a2
# ╟─5a3373e4-5643-482b-9f35-ebdb6fbc7559
# ╟─4d0490c4-b27e-423d-9298-35a60a3896c9
# ╟─0f37f95c-3122-4e2f-8bfd-65309f9182f6
# ╟─934f9f71-a7a9-454a-b7f5-6419e1148227
# ╟─ac5c1519-83c8-42fc-81a5-4523d19eef23
# ╟─54c23d2e-9c4e-4fd3-aa3d-67feaf1f632b
# ╟─26de0c1c-db09-4f46-b23d-31ebbb6bf24b
# ╟─cbbfa4b2-7de1-4ad5-9f75-af6468ae8415
# ╟─e9803209-c3b7-4c21-9a35-86b69c0995ca
# ╟─394910ab-987a-4678-9cc0-e2020511b9d2
# ╟─41a5afbb-c6ec-43b0-847d-9b7929f3d1b7
# ╟─4c08134f-9a0a-4a3a-90c7-fd9c9dfcbb38
# ╟─bf9bda39-d081-4a68-8091-ae82f8ca8f35
# ╟─10be7f37-3526-48ca-b56f-d81a949f2fee
# ╟─ee4830a9-4c0e-45b4-8fdc-74ec0e8d7fd6
# ╟─139c96a3-83da-4a5b-994b-f42bfaca83ab
# ╟─646c9815-bd63-4a31-ad04-e5d85df9efd0
# ╟─397269f9-ff8f-41cd-af41-5aefab6b2dba
# ╟─05d47e31-9ab0-4abb-bd77-019259b70437
# ╟─47a405ca-963a-40d0-a116-45bf6cd848ca
# ╟─32c3e468-7353-4e4f-b1b0-65a1953d174d
# ╟─8dfc789e-8629-479e-86f9-32bfea9b4065
# ╟─d989fca3-a1a6-416a-928e-31ca6e166341
# ╟─01a12e9e-9dcc-43bb-83e5-3ca2042c75e4
# ╟─be267de9-2a9d-4b99-ae45-71cbcd53c41d
# ╟─0ed966e8-a242-4e96-b976-7174139cb279
# ╟─187aa1ec-fa10-4902-8bcd-72c86e391caf
# ╟─fc0b6698-c6a1-4315-8e5a-e7bc5234bd63
# ╟─12f8c22c-1c6e-4bf1-9b46-cf25a12b91de
# ╟─470f18e5-e0e1-411e-be46-df3cd5a43d6c
# ╟─c39a870b-fd23-4dab-8f53-306e3846c5d3
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─c363d75a-adc3-49fb-ba3c-bb4b870dd7db
# ╠═b29505b8-6fda-4513-b1f2-59f78bd5d2c4
# ╠═1d5093fc-9fd7-4976-8768-064e323c3f31
# ╠═640efe26-b14b-43fe-b926-f7e749461b9d
# ╠═39b4fb9a-8776-47dd-8cb9-9b754e79d40e
# ╠═820a5289-ef20-4343-aeaf-2a8bb6cddcb2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
