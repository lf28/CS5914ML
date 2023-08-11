### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ f7ad60c5-aaa1-46b0-9847-5e693503c5ca
begin
	using PlutoTeachingTools
	using PlutoUI
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using Distributions, LinearAlgebra
	using StatsPlots
	using LogExpFunctions
	# using Statistics
	using StatsBase
	using LaTeXStrings
	using Latexify
	using HypertextLiteral
	using Random
	using GLM
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

# ╔═╡ c0e7844b-111f-469a-9d6e-33751a441adb
begin
	function true_f(x)
		-5*tanh(0.5*x) * (1- tanh(0.5*x)^2)
	end
	Random.seed!(100)
	x_input = [range(-6, -1.5, 10); range(1.5, 6, 8)]
	y_output = true_f.(x_input) .+ sqrt(0.05) * randn(length(x_input))
end;

# ╔═╡ 945acfa0-9fde-4635-b87e-7f7f026da8bb
plt1 = let
	scatter(x_input, y_output, label="", xlim=[-13, 13], ylim=[-5, 5], xlabel=L"x", ylabel=L"y")	
	plot!(true_f, lw=2, framestyle=:default,  lc=:red, ls=:solid,label=L"f(x)", xlabel=L"x", ylabel=L"y")
end;

# ╔═╡ 3da83a5f-be41-47a1-bad1-781e59658f95
begin
	function poly_expand(x; order = 2) # expand the design matrix to the pth order
		n = length(x)
		return hcat([x.^p for p in 0:order]...)
	end
end

# ╔═╡ 0cae3fc5-19a3-4143-aebc-e9484f6253ba
begin
	Random.seed!(123)
	w = randn(length(x_input)) ./ 2
	b = range(-18, 22, length(x_input)) |> collect
	Φ = tanh.(x_input .- w' .* b')
end;

# ╔═╡ 18c3484f-c219-4a98-95c7-81c87f5c5ff0
begin
	poly_order = 6
	# Φ = poly_expand(x_input; order = poly_order)
	freq_ols_model = lm(Φ, y_output);
	# apply the same expansion on the testing dataset
	x_test = -10:0.1:10
	Φₜₑₛₜ = tanh.(x_test .- w' .* b')
	tₜₑₛₜ = true_f.(x_test)
	# predict on the test dataset
	βₘₗ = coef(freq_ols_model)
	pred_y_ols = Φₜₑₛₜ * βₘₗ 
end;

# ╔═╡ 43f36576-b470-4fcf-935a-dca80c2f80f8
plt2 = let
	# plot(x_test, tₜₑₛₜ, linecolor=:black, ylim= [-3, 3], lw=2, linestyle=:dash, lc=:gray,framestyle=:default, label="true signal")
	scatter(x_input, y_output, label="", xlim=[-13, 13], ylim=[-5, 5], xlabel=L"x", ylabel=L"y")	
	plot!(x_test, pred_y_ols, linestyle=:solid, lc =:blue, lw=2, xlabel=L"x", ylabel=L"y", legend=:topright, label=L"f(x)")
end;

# ╔═╡ 94ef0651-e697-4806-868a-bce137d8437e
plt=plot(plt1, plt2, layout=(1,2), size=(600,300));

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
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.86"
GLM = "~1.8.2"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.18"
LogExpFunctions = "~0.3.23"
Plots = "~1.38.8"
PlutoTeachingTools = "~0.2.8"
PlutoUI = "~0.7.50"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "308c6c125a550c3977a20a470dd56f4531158754"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8bc0aaec0ca548eb6cf5f0d7d16351650c1ee956"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.2"
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
git-tree-sha1 = "db40d3aff76ea6a3619fdd15a8c78299221a2394"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.97"

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
git-tree-sha1 = "0b3b52afd0f87b0a3f5ada0466352d125c9db458"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.2.1"

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

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "97829cfda0df99ddaeaafb5b370d6cab87b7013e"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.3"

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
git-tree-sha1 = "2613d054b0e18a3dea99ca1594e9a3960e025da4"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.7"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "0ec02c648befc2f94156eaef13b0f38106212f3f"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.17"

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

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

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
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "832afbae2a45b4ae7e831f86965469a24d1d8a83"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.26"

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
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

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
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "8cc7a5385ecaa420f0b3426f9b0135d0df0638ed"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.2"

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
# ╟─c0e7844b-111f-469a-9d6e-33751a441adb
# ╟─94ef0651-e697-4806-868a-bce137d8437e
# ╟─945acfa0-9fde-4635-b87e-7f7f026da8bb
# ╟─43f36576-b470-4fcf-935a-dca80c2f80f8
# ╟─3da83a5f-be41-47a1-bad1-781e59658f95
# ╟─0cae3fc5-19a3-4143-aebc-e9484f6253ba
# ╟─18c3484f-c219-4a98-95c7-81c87f5c5ff0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
