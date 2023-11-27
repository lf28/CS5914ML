### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ f7ad60c5-aaa1-46b0-9847-5e693503c5ca
begin
	using PlutoTeachingTools
	using PlutoUI
	# using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	# using Distributions, LinearAlgebra
	# using StatsPlots
	# using LogExpFunctions
	# using Statistics
	# using StatsBase
	using LaTeXStrings
	using Latexify
	using HypertextLiteral
	# using Random
	# using GLM
end

# ╔═╡ 6b33f22b-02c4-485a-a8e0-8329e31f49ea
TableOfContents()

# ╔═╡ 263003d7-8f36-467c-b19d-1cd92235107d
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ e03e585c-23ab-4f1b-855f-4dcebee6e63c
bayes_figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/bayes/";

# ╔═╡ fdc69b6d-247c-49f4-8642-0345281c2370
ChooseDisplayMode()

# ╔═╡ 45b8e008-cfdc-11ed-24de-49f79ef9c19b
md"""

# CS5914 Machine Learning Algorithms


#### Probability theory 2
###### Probabilistic graphical models
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ db1f6632-f894-4ce7-a5b4-493158a13860
md"""

# More probability Theory 

"""

# ╔═╡ 7416236c-5ff0-4eb8-b448-e50acb0c016b
md"""
## Recap: joint distributions


A **joint distribution** over a set of random variables: ``X_1, X_2, \ldots, X_n`` 
* specifies a real number for each assignment (or outcome): 


```math
\begin{equation} P(X_1= x_1, X_2=x_2,\ldots, X_n= x_n) = P(x_1, x_2, \ldots, x_n) \end{equation} 
```

* Must still statisfy

$P(x_1, x_2, \ldots, x_n) \geq 0\;\; \text{and}\;\;  \sum_{x_1, x_2, \ldots, x_n} P(x_1, x_2, \ldots, x_n) =1$ 

##

**An example** joint distribution of temperature (``T``) and weather (``W``): ``P(T,W)``
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

# ╔═╡ 48484810-1d47-473d-beb3-972ca8b934d5
md"""
## Two probability rules


There are only **two probability rules** 

> #### Rule 1: *sum rule* 

> #### Rule 2: *product rule* (*a.k.a* chain rule)




"""

# ╔═╡ 559b0936-a051-4f2e-b0ca-81becd631bce
md"""
## Probability rule 1: sum rule



> $$\large P(X_1) = \sum_{x_2}P(X_1, X_2=x_2);\;\; P(X_2) = \sum_{x_1}P(X_1=x_1, X_2),$$

* ``P(X_1), P(X_2)`` are called **marginal probability distribution**

"""

# ╔═╡ 1a930c29-0b6e-405b-9ada-08468c69173c
md"""
## Example -- sum rule

"""

# ╔═╡ ddc281f3-6a90-476a-a9b9-34d159cca80a
Resource(figure_url * "/figs4CS5010/sumrule_.png", :width=>800, :align=>"left")

# ╔═╡ 05e3c6d0-f59b-4e1e-b631-11eb8f8b0945
md"""
## Probability rule 2: chain rule


Chain rule (*a.k.a* **product rule**)

> $$\large P(X, Y) = P(X)P(Y|X);\;\; P(X, Y) = P(Y)P(X|Y)$$

  * the chain order doesn't matter
  * joint distribution factorised as a product
    * between marginal and conditional

"""

# ╔═╡ c0e0f03f-087e-418a-9c5b-dd96baaf5719
md"""
## Example -- chain rule

> $$\large P(D, W) = P(W)P(D|W)$$

"""

# ╔═╡ 67791314-3fbb-44a5-af3a-16b1de95d351
Resource(figure_url * "/figs4CS5010/prodrule.png", :width=>800, :align=>"left")

# ╔═╡ bb2c9a6b-1957-4360-9fe1-6163d57f668b
md"""
## Example -- chain rule

"""

# ╔═╡ 007784a6-1b31-4ec5-8b97-e46d532ae259
Resource(figure_url * "/figs4CS5010/prodrule1.png", :width=>800, :align=>"left")

# ╔═╡ 1d668202-ca81-4934-824f-f340a936b375
md"""
## Example -- chain rule

"""

# ╔═╡ ec913363-60d7-4212-994b-29ad0373758d
Resource(figure_url * "/figs4CS5010/prodrule2.png", :width=>800, :align=>"left")

# ╔═╡ dc66e4b8-f9d0-4224-8c29-fb9fd733b8b5
md"""
## Example -- chain rule

"""

# ╔═╡ 275da2b1-fa16-4c57-8639-3468a2438a0c
Resource(figure_url * "/figs4CS5010/prodrule3.png", :width=>800, :align=>"left")

# ╔═╡ 32bd86a4-0396-47f5-9025-6df83268850e
md"""

## Product rule: generalisation


Chain rule for higher-dimensional random variables

* for three random variables
$P(x_1, x_2, x_3) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2)$

* for any ``n``:

$P(x_1, x_2, \ldots, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2) \ldots=\prod_{i=1}^n P(x_i|x_1, \ldots, x_{i-1})$

> **Chain rule**: chain distributions together
"""

# ╔═╡ e12f0e44-2b7d-4075-9986-6322fb1498c4
md"""

## Revisit Bayes' rule



##### *Bayes' rule* is a direct result of the two rules (based on two r.v.s)

* ``H``: hypothesis and ``\mathcal{D}``: observation

* numerator: product rule
* denominator: sum rule

"""

# ╔═╡ fff67cca-de10-420f-b96e-0f2b9ba7d82b
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/bayesrule.png' width = '450' /></center>"

# ╔═╡ 6db0a9fa-0137-413e-b122-24a51c4319e0
md"""

## Recall: independence

Random variable $X,Y$ are independent, if 

> $$\large \forall x,y : P(x,y) = P(x)P(y)$$

* the joint is formed by the product of the marginals

Another equivalent definition

> $$\large \forall x,y: P(x|y) = P(x)$$
  * intuition: knowing (conditional on $Y=y$) does not change the probability of ``X``

"""

# ╔═╡ c594b3e3-1292-4296-be28-96633162bf8c
# md"""

# Chain rule + independence assumption:

# ```math
# P(X, Y) = P(Y) P(X|y)
# ```

# """

# ╔═╡ 8afac55e-44f9-4127-b16a-969560307f90
md"""

## Conditional independence

``X`` is conditionally independent of ``Y`` given ``Z``, if and only if 


> $$\large \forall x,y,z: P(x,y|z) = P(x|z)p(y|z)$$

or equivalently (**more useful**), if and only if 

> $$\large \forall x,y,z: P(x|z, y) = P(x|z)$$

* intuition: knowing ``z``, ``x`` is no longer influenced by ``y`` (therefore independence): ``p(x|z, \cancel{y})``


## Conditional independence

``X`` is conditionally independent of ``Y`` given ``Z``, if and only if 


> $$\large \forall x,y,z: P(x,y|z) = P(x|z)p(y|z)$$

or equivalently (**more useful**), if and only if 

> $$\large \forall x,y,z: P(x|z, y) = P(x|z)$$

* intuition: knowing ``z``, ``x`` is no longer influenced by ``y`` (therefore independence): ``p(x|z, \cancel{y})``


> Conditional independence is denoted as 
> 
> $X\perp Y\vert Z$





## Conditional and Unconditional independence

Note that (*marginal or unconditional independence*) is a specific case of conditional independence (the condition is an empty set), *i.e.*

```math
P(X,Y|\emptyset) = P(X|\emptyset)P(Y|\emptyset)
```

* *i.e.* ``X\perp Y \vert \emptyset``
* the independence relationship is unconditional



"""

# ╔═╡ 5bcfc1a8-99d4-4b48-b45c-4c8d0c5acae4
md"""

## Example

Consider three r.v.s: **Traffic, Umbrella, Raining**


Based on *chain rule* (or product rule)

$$P(\text{Rain},\text{Umbrella}, \text{Traffic}) = P(\text{Rain})  P(\text{Traffic}|\text{Rain}) P(\text{Umbrella}|\text{Traffic}, \text{Rain})$$


"""

# ╔═╡ 6dc34d66-366f-4e1a-9b73-a74c07e58feb
md"""

## Example

Consider 3 r.v.s: **Traffic, Umbrella, Raining**


Based on *chain rule* (or product rule)

$$P(\text{Rain},\text{Umbrella}, \text{Traffic}) = P(\text{Rain})  P(\text{Traffic}|\text{Rain}) P(\text{Umbrella}|\text{Traffic}, \text{Rain})$$


The last term can be simplified based on **conditional independence assumption**

$$P(\text{Umbrella}|\cancel{\text{Traffic}}, {\text{Rain}}) = P(\text{Umbrella}|\text{Rain})$$

* knowing it is raining, Umbrella no longer depends on Traffic

* or in CI notation: ``T \perp U|R``



"""

# ╔═╡ 12751639-7cdc-4e39-ba72-30ee2fe2cd49
md"""

## Example (Coin switch problem)

Recall the coin switching problem

> Two coins 
> * one fair ``p_1= 0.5`` 
> * the other is bent with bias $p_2= 0.2$ 
> The player switches to the bent coin after some **unknown** number of switches
> When did he switch?

Based on *chain rule* (or product rule)

$$P({S},\mathcal{D}) = P(S)  P(\mathcal{D}|S)$$

* ``\mathcal{D} = [d_1, d_2, \ldots, d_n]``
Then applies **conditional independence assumption**

$$P(\mathcal{D}|S) = P(d_1|S)P(d_2|S) \ldots P(d_n|S)$$

* conditioned on the switching time ``S``, the tosses are independent 

"""

# ╔═╡ 0198a401-ed47-4f5c-8aca-441cccff472b
md"""

# Probabilistic graphical models
"""

# ╔═╡ 532a9f66-b04a-446a-becf-d74759bcf750
md"""
## Probabilistic graphical models


Probabilistic graphical models are very useful probabilistic modelling tool

* to represent the full joint ``P(X_1, X_2, \ldots, X_n)``
* and to represent conditional independence graphically


## Probabilistic graphical models


Probabilistic graphical models are very useful probabilistic modelling tool

* to represent the full joint ``P(X_1, X_2, \ldots, X_n)``
* and to represent conditional independence graphically



"""

# ╔═╡ 7b2e5d23-2775-484c-aca3-bd05ff889331
TwoColumn(md"""

\

We consider a specific kind: **directed graphical model** 
* also known as Bayes' network
* a directed acyclic graph (DAG)
  * **node**: random variables 
  * **edges**: relationship between random variable



""", html"""<p align="center">For example, the cough example</p><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/cough_bn.png" width = "100"/></center>
""")

# ╔═╡ 9da0bf21-3d2f-4df0-a8b3-3891dd665c7f
md"""

## Digress: directed graphs

A **directed** graph is a a graph with **directed** edges
  * direction matters: $(X, Y)$ are not the same as $(Y, X)$
  * **asymmetric relationship**: 
    * e.g. parent-to-child relationship (the reverse is not true)
    

``\textbf{parent}(\cdot)`` returns the set of parent nodes, e.g. $$\text{parent}(Y) = \{X, Z\}$$
"""

# ╔═╡ 24f91696-3c70-472c-89db-fc4732c28677
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/directedGraph.png" width = "400"/></center>"""

# ╔═╡ 41255527-96d2-41bf-92b4-8b4a9bebab24
md"""
## Digress: directed acyclic graph (DAG)

A **d**irected **a**cyclic **g**raph (**DAG**) is a directed graph **without** cycles 
  * a cycle: directed path starts and ends at the same node
"""

# ╔═╡ 00a0c582-34b2-413d-bef2-7a7572f48340
md"""

**For example**, the following is **NOT** a DAG
* cycles are NOT allowed: $X\Rightarrow Y_1\Rightarrow X$

"""

# ╔═╡ 144eb24c-1b4a-44d2-bd25-33b68798c184
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/diceExampleDCG.png" width = "300"/></center>"""

# ╔═╡ a578a1a7-0f42-492c-8226-b8e09e92506f
md" However, multiple paths are allowed, *e.g.*"

# ╔═╡ 288849aa-35c5-426f-89aa-76d11a5801e9
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/diceExampleDAG.png" width = "300"/></center>"""

# ╔═╡ 20c95319-86a7-477d-93c8-adf0dbdbb4c2
md"""
  * two possible paths from $X$ to $Y_2$: $X\Rightarrow Y_2$ and $X \Rightarrow Y_1 \Rightarrow Y_2$
  * still **acyclic** though
"""

# ╔═╡ 275c155d-c4c5-472a-8613-c549706815b0
md"""

## Node types

"""

# ╔═╡ 2f5af892-249f-40c8-a2a2-0bc7acc7f000
md"""

We introduce three types of nodes

* **unshaded**: unobserved **random variable** (the hidden cause)
* **shaded**: observed **random variable** (coughing)
* **dot**: fixed parameters/input, non-random variables
  * *e.g.* the input predictors ``\mathbf{x}`` for linear regression
  * or fixed hyperparameters 
"""

# ╔═╡ 7b9577e1-e1ce-44a9-9737-a60061ad0dc7
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/notations.png' width = '200' /></center>"

# ╔═╡ 41ee5127-4390-4fb8-b41f-9856346877cf
md"""

## Directed graphical model: a DAG with local distributions 


A Directed graphical model is 

> **Directed Acyclic Graph (DAG)** + local prob. distributions, $P(X_i|\text{parents}(X_i))$
  

* **Local distributions**: $P(X_i|\text{parent}(X_i))$, one $P$ for each r.v. $X_i$


"""

# ╔═╡ eda02250-b88d-45a1-9430-114e3d60f72a
TwoColumn(md"""


**For example**, the cough example

* ``H`` is with ``P(H)``
* ``\texttt{Cough}``: ``P(\texttt{Cough}|H=h)``
""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/cough_bn2.svg" height = "180"/></center>
""")

# ╔═╡ be4b66ed-dc94-4450-97dd-a57175913b20
md"""
## Graphical model -- factoring property

Graphical model in essence:

* a **graphical representation** of the **full joint distribution** & the _conditional independent_ relationships

!!! important "Factoring property & chain rule"
	Joint distribution factorises as the product of CPTs: 
	
	$\large P(X_1, X_2,\ldots, X_n) = \prod_{i=1}^n P(X_i|\text{parent}(X_i))$

"""

# ╔═╡ c7217173-3b9e-439a-ad59-6fdea81bb12c
md"""

## Example - Traffic, Rain, Umbrella

Recall the **Traffic, Umbrella, Raining** example


The full joint (based on the **conditional independence assumption**) becomes 

$$P(\text{Rain},\text{Umbrella}, \text{Traffic}) = P(\text{Rain})  P(\text{Traffic}|\text{Rain}) P(\text{Umbrella}|\text{Rain})$$

The corresponding graphical model is
* just a graphical representation of the r.h.s
"""

# ╔═╡ 4c4f8078-b3ed-479f-9c0b-35c9e81fe44d
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/rtu_bn.png" width = "300"/></center>
"""

# ╔═╡ 33a8d7e1-cb9b-48c8-9b63-7a817c0497b2
md"""

## Example -- coin guess


> There are two coins (with biases ``0.5`` and ``0.99`` respectively) in an urn
> * you randomly pick one and toss it 3 times 



"""

# ╔═╡ 14cc0014-980f-482a-9c1d-df20c383d7c3
TwoColumn(html"""<br/><br/><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/coin_choice.png" width = "400"/></center>""", md"""

Coin Choice: $P(C)$

|``C``   | ``P(C=c)`` |
| --- | ---  | 
| 1   | 0.5 | 
| 2   | 0.5 | 

Observation ``Y_i``: $P(Y_i|C)$

|``C``   | ``\texttt{head}`` | ``\texttt{tail}`` |
| ---| ---   | ---     |
| 1  | 0.5 | 0.5 |
| 2  | 0.01 | 0.99 |

* each row is a valid conditional distributions 


""")

# ╔═╡ b96cc13e-5fd9-4fa1-a9bc-4ea7b22c07a3
md"""

## Plate notation 

##### for repetitive random variables

\

Note that coin tosses are repeated, *i.e.* $P(Y_i|C)$ for $i =1,2,3$ are shared


"""

# ╔═╡ a94dac61-af9b-4820-bc2a-812a282a5603
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/plate_coin.png" width = "250"/></center>"""

# ╔═╡ 8044aa77-b299-4874-b90b-ee36fb8d374e
md"""

## Some graphical model patterns

There are three patterns in graphical models

* #####  common cause pattern
* ##### chain pattern
* ##### collider pattern

> All graphical models are **composition** of the three patterns



## Common cause pattern (*a.k.a.* tail to tail)

"""

# ╔═╡ ff9b9e50-f11f-4153-9dad-9b71ae82d2a4
TwoColumn(md"""

\

``Y`` is the common cause of ``X`` and ``Z``


* also known as tail to tail (from ``Y``'s perspective)
* *e.g.* the unknown bias is the common cause of the two tossing realisations

""",  html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/dag_common.png" width = "500"/></center>""")

# ╔═╡ 09464af5-29c3-4b26-a9a9-5fdc0b6cdaa6

md"""
## Common cause pattern (*a.k.a.* tail to tail)
"""

# ╔═╡ 8e7224f1-65bc-46a9-b4dc-13584ce7277d
TwoColumn(
md"""

It can be shown that 

```math
X \perp Z |Y\;\; \text{or}\;\;P(X,Z|Y) =P(X|Y)P(Z|Y)
```


But **marginally** not independent 

```math
X \not\perp Z|\emptyset\;\; \text{or}\;\;P(X,Z) \neq P(X)P(Z)
```

""",

html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/dag_common.png" width = "500"/></center>"""

	
)

# ╔═╡ 07105298-06b4-4a60-938c-914809b13705
TwoColumn(
md"""
\

**Example**: "Vaccine" effectiveness 
* common cause: whether a vaccine is effective (**Y**)
* ``X, Z`` nodes: two patients' treatment outcomes after receiving the vaccine (**X**, **Z**)
""",

html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/vaccine.svg" width = "250"/></center>"""

	
)

# ╔═╡ 588eec3d-fc0b-4384-ac9b-bb8d8816b184
md"""
## Chain pattern (*a.k.a.* tip to tail)

> ##### ``X`` impacts ``Z`` _via_ ``Y``
> * models **indirect** influence 

"""

# ╔═╡ 450de0ac-e2a9-4e0a-b542-72534c751fa2
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/dag_seq.png" width = "400"/></center>"""

# ╔═╡ fb1dcc57-b321-4c26-bd9d-2390783662f6
md"""

## Chain pattern (*a.k.a.* tip to tail)
"""

# ╔═╡ 59e98d81-aef0-4e96-95e0-eb7e464b1361
TwoColumn(

md"""

It can be shown 


```math
X \perp Z|Y \;\; \text{or}\;\;P(X,Z|Y) = P(X|Y)P(Z|Y)

```

but marginally **not** independent 

```math
X \not\perp Z|\emptyset \;\; \text{or}\;\;P(X,Z) \neq P(X)P(Z)

```

""",

	
html"""<br/><br/><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/dag_seq.png" width = "500"/></center>"""
	
)

# ╔═╡ c6653de5-1542-44a6-9fcb-2e9bd091f2bb
md"""

**Example**: `Rain`, `Floor_is_Wet`, `Slip`


```math
\begin{align}
{\text{Rain}} \rightarrow {\text{Floor is Wet}}  \rightarrow {\text{Slip}} 
\end{align}
```

* add *Injured* ?
"""

# ╔═╡ 9fafd041-2d42-4bf1-a17a-c31deb23d97d
md"""

## A classic example -- chain case

##### *Voting for Trump* does not *directly* make you more likely to *die*!
* but there is a hidden intermediate cause, *i.e.* vaccination rate
"""

# ╔═╡ 3db2f0f1-d8b1-4483-af5e-71889be766d9
md"""

A suitable model

```math
\text{Vote For Trump} \rightarrow \text{Vaccinated} \rightarrow \text{Die Out of COVID}

```

"""

# ╔═╡ 653ef7c0-8c52-4e7c-b402-e8269fa6dc8b
TwoColumn(html"""<center><img src="https://static01.nyt.com/images/2021/09/27/multimedia/27-MORNING-sub3-STATE-DEATH-VOTE-PLOT/27-MORNING-sub3-STATE-DEATH-VOTE-PLOT-superJumbo.png?quality=75&auto=webp" width = "320"/></center>
""", html"""<center><img src="https://static01.nyt.com/images/2021/09/27/multimedia/27-MORNING-sub2-STATE-VAX-VOTE-PLOT/27-MORNING-sub2-STATE-VAX-VOTE-PLOT-superJumbo-v2.png?quality=75&auto=webp" width = "320"/></center>
""")

# ╔═╡ 94767935-df48-4466-9454-aa000217ddda
md"""
## Common effect (*a.k.a.*  collider, or tip to tip)

"""

# ╔═╡ e01ac6d8-a7fa-4fc4-bdca-b58cc2ff920d
TwoColumn(md"""



\
\
\

``Y`` is the joint **effect** of some combination of *independent* causes ``X`` and ``Z``

""", 
	
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/dag_collider.png" width = "500"/></center>""")

# ╔═╡ e0adb7b4-1984-4002-a411-6a7fa2aff1dc
md"""
## Common effect (*a.k.a.*  collider, or tip to tip)

"""

# ╔═╡ a8ae1abc-8104-4bca-9fd4-18f5aa48dba5
TwoColumn(md"""

**Quite the opposite** to the previous two cases, 
* ``X, Z`` are marginally independent 
* but not conditionally independent (explain away)


```math
X\perp Z|\emptyset\;\; \text{or} \;\; P(X, Z)= P(X)P(Z)

```

```math
X\not \perp Z|Y\;\; \text{or} \;\; P(X, Z|Y)\neq P(X|Y)P(Z|Y)
```
```math
\text{or } P(X|Z, Y)\neq P(X|Y) 
```

""", 
	
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/dag_collider.png" width = "500"/></center>""")

# ╔═╡ 93f8d325-f007-4849-8b16-e77ab31088e7
TwoColumn(md"""
\

**Example**: Exam performance
* ``X``: your knowledge of the subject
* ``Z``: exam's difficulty 
* ``Y``: exam grade 

*Marginally,* `Knowledge` ``\perp`` `Exam_Difficulty` 



""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/examdiff.svg" width = "300"/></center>""")

# ╔═╡ 6e320e26-a270-4c21-8e74-a1c37de41ca9
md"""
*Conditional* on the effect `Grade`, they become dependent 

```math
\begin{align}
P(\texttt{Knowledge}=good|\texttt{Grade}=fail,\; \texttt{Exam\_diff}=easy)\; {\Large\neq}\; \\ P(\texttt{Knowledge}=good|\texttt{Grade}=fail, \texttt{Exam\_diff}=hard)
\end{align}
```


* exam is easy ``\Rightarrow`` `knowledge` is bad
* exam is difficult ``\Rightarrow`` `knowledge` is probably not that bad
"""

# ╔═╡ 7d48dff8-9b53-462a-bbf4-4a9cd4d51be9
md"""

## Summary


* Two probability rules: sum and chain rule


* To form a probabilistic model, it is easier to think generatively via chain rule

$$P(H,\mathcal{D}) = P(H) P(\mathcal{D}|H)$$


* Bayes' rule provides a way to do reverse inference 


* Probabilistic graphical model is a useful modelling tool


"""

# ╔═╡ 61e3f627-5752-48e3-9100-3c96575f1805
md"""

## Appendix
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
Latexify = "~0.15.18"
PlutoTeachingTools = "~0.2.8"
PlutoUI = "~0.7.50"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "e48e5505480980060ed6496934e2cd51576a4662"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0592b1810613d1c95eeebcd22dc11fba186c2a57"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.26"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

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
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

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
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "a38e7d70267283888bc83911626961f0b8d5966f"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.9"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

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

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─f7ad60c5-aaa1-46b0-9847-5e693503c5ca
# ╟─6b33f22b-02c4-485a-a8e0-8329e31f49ea
# ╟─263003d7-8f36-467c-b19d-1cd92235107d
# ╟─e03e585c-23ab-4f1b-855f-4dcebee6e63c
# ╟─fdc69b6d-247c-49f4-8642-0345281c2370
# ╟─45b8e008-cfdc-11ed-24de-49f79ef9c19b
# ╟─db1f6632-f894-4ce7-a5b4-493158a13860
# ╟─7416236c-5ff0-4eb8-b448-e50acb0c016b
# ╟─48484810-1d47-473d-beb3-972ca8b934d5
# ╟─559b0936-a051-4f2e-b0ca-81becd631bce
# ╟─1a930c29-0b6e-405b-9ada-08468c69173c
# ╟─ddc281f3-6a90-476a-a9b9-34d159cca80a
# ╟─05e3c6d0-f59b-4e1e-b631-11eb8f8b0945
# ╟─c0e0f03f-087e-418a-9c5b-dd96baaf5719
# ╟─67791314-3fbb-44a5-af3a-16b1de95d351
# ╟─bb2c9a6b-1957-4360-9fe1-6163d57f668b
# ╟─007784a6-1b31-4ec5-8b97-e46d532ae259
# ╟─1d668202-ca81-4934-824f-f340a936b375
# ╟─ec913363-60d7-4212-994b-29ad0373758d
# ╟─dc66e4b8-f9d0-4224-8c29-fb9fd733b8b5
# ╟─275da2b1-fa16-4c57-8639-3468a2438a0c
# ╟─32bd86a4-0396-47f5-9025-6df83268850e
# ╟─e12f0e44-2b7d-4075-9986-6322fb1498c4
# ╟─fff67cca-de10-420f-b96e-0f2b9ba7d82b
# ╟─6db0a9fa-0137-413e-b122-24a51c4319e0
# ╟─c594b3e3-1292-4296-be28-96633162bf8c
# ╟─8afac55e-44f9-4127-b16a-969560307f90
# ╟─5bcfc1a8-99d4-4b48-b45c-4c8d0c5acae4
# ╟─6dc34d66-366f-4e1a-9b73-a74c07e58feb
# ╟─12751639-7cdc-4e39-ba72-30ee2fe2cd49
# ╟─0198a401-ed47-4f5c-8aca-441cccff472b
# ╟─532a9f66-b04a-446a-becf-d74759bcf750
# ╟─7b2e5d23-2775-484c-aca3-bd05ff889331
# ╟─9da0bf21-3d2f-4df0-a8b3-3891dd665c7f
# ╟─24f91696-3c70-472c-89db-fc4732c28677
# ╟─41255527-96d2-41bf-92b4-8b4a9bebab24
# ╟─00a0c582-34b2-413d-bef2-7a7572f48340
# ╟─144eb24c-1b4a-44d2-bd25-33b68798c184
# ╟─a578a1a7-0f42-492c-8226-b8e09e92506f
# ╟─288849aa-35c5-426f-89aa-76d11a5801e9
# ╟─20c95319-86a7-477d-93c8-adf0dbdbb4c2
# ╟─275c155d-c4c5-472a-8613-c549706815b0
# ╟─2f5af892-249f-40c8-a2a2-0bc7acc7f000
# ╟─7b9577e1-e1ce-44a9-9737-a60061ad0dc7
# ╟─41ee5127-4390-4fb8-b41f-9856346877cf
# ╟─eda02250-b88d-45a1-9430-114e3d60f72a
# ╟─be4b66ed-dc94-4450-97dd-a57175913b20
# ╟─c7217173-3b9e-439a-ad59-6fdea81bb12c
# ╟─4c4f8078-b3ed-479f-9c0b-35c9e81fe44d
# ╟─33a8d7e1-cb9b-48c8-9b63-7a817c0497b2
# ╟─14cc0014-980f-482a-9c1d-df20c383d7c3
# ╟─b96cc13e-5fd9-4fa1-a9bc-4ea7b22c07a3
# ╟─a94dac61-af9b-4820-bc2a-812a282a5603
# ╟─8044aa77-b299-4874-b90b-ee36fb8d374e
# ╟─ff9b9e50-f11f-4153-9dad-9b71ae82d2a4
# ╟─09464af5-29c3-4b26-a9a9-5fdc0b6cdaa6
# ╟─8e7224f1-65bc-46a9-b4dc-13584ce7277d
# ╟─07105298-06b4-4a60-938c-914809b13705
# ╟─588eec3d-fc0b-4384-ac9b-bb8d8816b184
# ╟─450de0ac-e2a9-4e0a-b542-72534c751fa2
# ╟─fb1dcc57-b321-4c26-bd9d-2390783662f6
# ╟─59e98d81-aef0-4e96-95e0-eb7e464b1361
# ╟─c6653de5-1542-44a6-9fcb-2e9bd091f2bb
# ╟─9fafd041-2d42-4bf1-a17a-c31deb23d97d
# ╟─3db2f0f1-d8b1-4483-af5e-71889be766d9
# ╟─653ef7c0-8c52-4e7c-b402-e8269fa6dc8b
# ╟─94767935-df48-4466-9454-aa000217ddda
# ╟─e01ac6d8-a7fa-4fc4-bdca-b58cc2ff920d
# ╟─e0adb7b4-1984-4002-a411-6a7fa2aff1dc
# ╟─a8ae1abc-8104-4bca-9fd4-18f5aa48dba5
# ╟─93f8d325-f007-4849-8b16-e77ab31088e7
# ╟─6e320e26-a270-4c21-8e74-a1c37de41ca9
# ╟─7d48dff8-9b53-462a-bbf4-4a9cd4d51be9
# ╟─61e3f627-5752-48e3-9100-3c96575f1805
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
