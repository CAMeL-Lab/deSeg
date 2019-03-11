# De-lexical Segmentation

Given a **grammar** describing the combinatorics of some (or all) closed class morphemes and an **analyzer** which uses only this grammar to generate all possible segmentation analyses allowing for any open class base, we train a **disambiguator** on a raw corpus to select the correct segmentation. For example, a simple grammar for English might allow for suffixal morpheme combinations such as '+ing', '+ing +s', and '+s', as attested in 'craving', 'happenings', and 'creates', respectively, though without any knowledge of open class base forms, our analyzer would provide at least three analyses of 'happenings': (1) 'happenings' (2) 'happening +s' (3) 'happen +ing +s'. Then the disambiguator would be charged with selecting the correct analysis. Obviously, knowledge of open class base forms would allow you to perform greedy, maximum matching such that a bad analysis like (1) is thrown out, but creating the open class resources necessary for this is expensive and not feasible for languages characterized by unstandardized spelling, diglossia, or otherwise wide dialectal variation. Hence, delexical segmentation can be a highly cost effective segmentation solution for morphologically rich languages as it relies on only a grammar of closed class forms that can be written in a few hours.

The analyzer however, will of course over generate, so the disambiguator must be able to choose the correct analysis with some degree of accuracy. Our tool considers three factors for unsupervised disambiguation: (1) fertility of the proposed base (number of unique combinations of morphemes that can be immediately pre/postpended to the proposed base) (2) length of the proposed base (if you greedily maximum match on the closed class morphemes, you are effectively minimizing the length of the base) (3) frequency count with which the proposed base occurs as a stand-alone word in the raw corpus (bases in many languages are highly likely to be free morphemes, meaning that they are likely to occur in the corpus as their own word). You can specify a preference over these factors from the command line during training, though the best performance seems to be ranking base fertility above base minimization above base count.

The analyzer included in this repository runs on Modern Standard Arabic (which is the language of *sample.in*, the white space/punctuation separated sample training data). Our goal was to segment all clitics, but we could not release the grammar we used for copyright reasons, so instead we release a free grammar that has been hackily accomodated in the analyzer to segment all affixes. The grammar was designed as a lexical tool, but we stripped the open class stems for this project. You can easily write your own grammar and tweak the Analyzer class to extend *deSeg* to the language of your choice. The disambiguator just expects the *get_possible_segmentations()* method of the analyzer to return a list of triples for each input word, where each triple is an analysis, comprised of a potentially empty list of prefixal elements, the base string, and a potentially empty list of suffixal elements.


## Quick Start

This is a short demo. For a full description of the various models you can train with *deSeg*, see Usage Options below.

### Prerequisites

* [Python 3](https://www.python.org/downloads/)
* [scipy](https://www.scipy.org)
* [camel_tools](https://camel-tools.readthedocs.io/en/latest/)

### Demo

Train the segmenter on unannotated sample data using the built-in, free grammar for Modern Standard Arabic.

```python deSeg.py -m train -t sample.in -g grammar.db -a built-in```

Run the trained segmenter in interactive mode.

```python deSeg.py -m interactive -c  disambiguator.sample.in.grammar.db.minBase1.pkl```

Apply the segmenter to the sample data.

```python deSeg.py -m apply -c disambiguator.sample.in.grammar.db.minBase1.pkl -T sample.in -o sample.out```

## Usage Options

In addition to specifying the relevant disambiguation models as discussed above, you can also specify the following options at the command line:

* ```-m``` Mode. *deSeg* can be run in ```train```, ```interactive```, or ```apply``` mode.
* ```-t``` Training file. This is raw data to be run through the analyzer to gather frequency statistics and train the disambiguator.
* ```-T``` Test file. This is the data which the trained disambiguator will tokenize.
* ```-o``` Output file. This is where the tokenized output will be written.
* ```-g``` Grammar. This is the location of the grammar database describing the combinatorics of closed class forms.
* ```-a``` Accomodated (sub)grammar(s). This option can be used to handle diglossia or other dialectal variation, and thus, it takes multiple arguments. We also use to it to handle the fact that the built-in grammar produces slightly differently formatted anaylises than the grammar this tool was originally developed with. Specify ```built-in``` to use the free grammar attached in this repository.
* ```-c``` Cached disambiguator model. This option specifies the location to look for a previously cached disambiguator model.
* ```-p``` Priority. This specifies the priority over factors affecting the disambiguator. ```f``` represents base fertility, ```b``` represents base length minimization, and ```c``` represents base count. The allowable options for priority are thus (1) ```fbc``` (1) ```fcb``` (1) ```bfc``` (1) ```bcf``` (1) ```cfb``` (1) ```cbf```. By defualt, ```fbc``` is used.
* ```-P``` Print options. ```most_frequent_tokens``` will print the most frequent tokens for each class (prefixal, base, suffixal) to standard output. ```ranked_tokenizations_by_word``` will print the ranking over possible tokenizations for every single word in the test file. This can only be used in apply mode.
* ```-d``` Debug. If True, this will print statistics for every token in every possible analyses when running in apply or interactive mode.
* ```-s``` Separator. This is the charactor appended to the end of prefixal elements or beginning of suffixal elements to signal how they attach to the base. The separator is '+' by default.
* ```-l``` Minimum Length of candidate base forms. Analyses with base forms of length less than this argument will only be considered if said base is the entire word.
* ```-M``` Multiple threads. This specifies how many threads to use in ```train``` mode. By defualt, *deSeg* will try to use 12 threads, each analyzing 1,000 word chunks of the vocabulary before consilidating the analyses to compute statistics for the disambiguator.

## Acknowledgments

*deSeg* was constructed at the New York University Abu Dhabi's [CAMeL Lab](https://nyuad.nyu.edu/en/research/centers-labs-and-projects/computational-approaches-to-modeling-language-lab.html) using the [Calima Star Analyzer](https://calimastar.abudhabi.nyu.edu/#/analyzer).
