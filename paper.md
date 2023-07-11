# Learning Without a Language of Thought

## Introduction

How do we acquire meaning?

This question has been at the center of much research in cognitive science over the last several decades, and yet there is still much that remains unknown. Part of the difficulty is that we still do not have a clear picture of how meaning is represented in the mind---the human brain, for all intents and purposes, is a black box.

Many models of acquisition rely on the **language of thought** (LOT), which provides an intuitive framework for understanding concept and word learning. At its core, the LOT hypothesis claims that there is a linguistic structure inherent to mental representation. For word learning, this bears out quite straightforwardly: words can be expressed as logical expressions composed of a finite set of semantic primitives which are 'core' to all learners. Language-of-thought word learning models approach acquisition as a statistical inference process, in which the learner iteratively builds mental representations of word meanings from this conceptual core.

LOT is an appealing framework for models of acquisition because it takes advantage of formal semantic analyses of meaning. This is especially useful for contexts like function words, where a word's `meaning' refers to its behavior as a logical and semantic *operator*. Acquisition models which utilize LOT have generally been successful at capturing human-like patterns of learning.

At the same time, neural language models (LMs hereafter) have also seen immense success in modeling language, despite utilizing an entirely different framework for representing meaning. To LMs, meaning is **distributional**: the surrounding context of a word determines how it is represented in the meaning space. These distributional meanings are learned from large amounts of linguistic data.

The empirical success of LMs cuts against the idea that humans build meanings from a set of primitives. This has been a large point of contention in the literature. Some (e.g. Bender and Koller) argue that distributional embedding spaces extracted from text alone cannot truly be conceieved of as representations of 'meaning' in any real way, while others (e.g. Merrill et al.) have shown that distributional models *are* capable of acquiring meaning because semantic information is implicitly encoded in textual data.

While all of this discussion is highly relevant to the study of language models, it is still an empirical question as to how *humans* acquire meaning. To this end, Chang and Bergen (2022) provide a useful framework for studying acquisition using language models. They posit that by comparing the acquisition *trajectories* of both humans and LMs, we can better understand the relationship between the two. They find that—on a broad set of words—the importance of various factors for acquisition greatly differs between humans and LMs, perhaps suggesting that the two utilize fundamentally distinct processes for meaning acquisition.

How, then, do neural LMs compare to LOT models? Here, we offer such a head-to-head comparison, by focusing on the acquisition of **temporal connectives**, a class of function words like \`before' and `while.' Temporal connectives are particularly interesting because they exhibit very particular developmental trajectories relative to one another. In particular, **semantically complex** connectives are learned later than logically simpler connectives.

Because the LOT conceives of meaning as the composition of semantic primitives, it is a natural framework within which to model temporal connective acquisition. Gorenstein et al. (2020) provide such a LOT model which accurately captures human-like developmental trajectories. This model is successful in large part because it explicitly represents this idea of semantic complexity: more complex meanings have lower prior production probabilities, leading to a bias towards simplicity during inference.

On the other hand, given the successes of distributional models of semantics---which do not explicitly encode any notion of complexity, or indeed, of compositionally---it is natural to ask whether such an encoding is even necessary. More simply put, does meaning acquisition rely on an explicit language of thought?

In the remainder of this paper, we first survey empirical and theoretical work on temporal connectives, and provide more rigorous semantic analyses of these function words (Section 2). In Section 3, we then detail Gorenstein et al. (2020)'s LOT model of acquisition, highlighting in particular its strength at encoding complexity biases. Sections 4 and 5 detail a rigorous head-to-head comparison between the LOT acquisition model and an LSTM model, focusing particularly on positive evidence from learning trajectories and negative evidence from sources of inaccuracy. Section 6 concludes.

## Why Are Connectives So Important?

I ran the experiments *before* I wrote the paper. Cher believes in life *after* love. We regularly have to describe events that are ordered in time. Across languages, speakers employ temporal connectives to express relative temporal relations. Indeed, these function words are so ubiquitous that many argue they are genuine examples of semantic universals.

Interestingly, however, temporal connectives are highly unique from an acquisition perspective: they are learned late, and they are learned in a highly specific order. In particular, Feagans (1980) notes that children tend to first acquire words communicating sequential order, like \`before' and \`after,' followed by words that communicate durational co-occurence like \`while,' and finally words like \`since' and \`until,' which communicate *both* order and duration.

And even with in these categories, learning is still somewhat differentiated. Clark (1971) and later work has found that \`before' is generally learned prior to \`after,' and similarly that \`since' is typically learned prior to \`until' in English. These phenomena are not just English-specific, either; Winskel (2003) reports similar patterns in Thai and Lisu.

### Theoretical Accounts

This poses an interesting problem for modeling connective acquisition---what is responsible for these differentiated trajectories? Almost all work in this area has centered around the idea of semantic complexity: temporal connectives vary in their \`complexity' under various formulations, and this variance gives rise to differentiated learning. In particular, the meanings of less \`complex' connectives are acquired faster, which is why we see e.g. young children who can comprehend \`before' but not \`since.'

In early work on the subject, Clark (1971) proposed a model called the **Semantic Features Acquisition theory.** Under this model, connective meanings are built out of hierarchical combinations of binary features; so, for example, \`before' is represented as $\langle+\text{time},-\text{simultaneous},+\text{prior}\rangle$ and \`after' is represented as $\langle+\text{time},-\text{simultaneous},-\text{prior}\rangle$. Then, Clark proposes that less complex combinations of features are easier to acquire; in particular, *positive* values are *less* complex. Thus, \`before' would be learned prior to \`after,' consistent with behavioral observations.

Clark's featural model, while promising, is ultimately hard to generalize to more nuanced temporal connectives (e.g. \`since') as well as other function words. Feagans (1980), however, builds on Clark (1971)'s notion of complexity-centric acquisition by turning to the formal semantics literature. In particular, Feagans notes that temporal connectives with longer logical representations tend to be learned later than those with short and straightforward representations.

With this in mind, the language-of-thought provides a very straightforward framework by which to measure `complexity.' Because LOT conceives of concepts as compositional expressions made up of a finite set of semantic primitives, semantic complexity can be understood within the context of these compositional expressions.

In Table~\ref{tab:meanings}, we list the logical expressions corresponding to the meaning representations for a set of five key English connectives. Meanings are defined on top of \`contexts' as lambda expressions with interval-based event representations. Each lambda expression takes as input an utterance time $t$ as well two events $A$ and $B$, where each event $E$ is an interval over time with start and end points $e_1$ and $e_2$. For example, `before' is represented as
\[\lambda\,A\,B\,t\,.\,a_1 < b_1.\]
This would return true in any context where $a_1 < b_1$.

## Using LOT to Model Connective Acquisition



## Using LMs to Model Connective Acquisition

### Methods

### Evaluation

### Results

## Discussion

