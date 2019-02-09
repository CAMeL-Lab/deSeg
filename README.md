# Greedy Tokenizer

Given an analyzer class with a method that returns all possible tokenizations of an input word, we train a greedy, unsupervised tokenization model to disambiguate these possible tokenizations. Tokenization depends only on statistics of observed possible tokenizations during training, i.e., context is not considered in this implementation. Thus, all word types will be deterministically tokenized regardless of context. The quality of the tokenization is completely dependent on the assumption that tokens frequently proposed by the analyzer are more likely to appear in the correct tokenization than tokens which are proposed less frequently.

The analyzer included in this repository runs on Modern Standard Arabic (which is the language of *sample.in*, the white space/punctuation separated sample training data). Our goal was to tokenize all clitics, but you can easily replace the analyzer and use the disambiguator to tokenize other pre/suffixal morphemes. The disambiguator just expects the *get_possible_tokenizations()* method of the analyzer to return a list of triples for each input word, where each triple is an analysis, comprised of a potentially empty list of prefixal elements, the base string, and a potentially empty list of suffixal elements.

We intend for this implementation to be adapted to handle under resourced languages where an analyzer of closed-class morphemes can be cheaply built, but there is no lexical information available. That is to say that we have or can quickly build a database of the possible affixes and associated morpho-syntactic behavior, but we don't have any information about the existance or combinatorics of open-class stems. An analyzer built on such a database would produce the set of analyses that includes any allowable combination of prefixes and suffixes that could be realized by a given word, but it would not be able to prune any of these as invalid because it would not be able to check if the remaining stem is a valid open class stem or not.

## Quick Start

This is a short demo. For a full description of the various models you can train with the Greedy Tokenizer, see Disambiguation Models below.

### Prerequisites

* [Python 3](https://www.python.org/downloads/)
* [scipy](https://www.scipy.org)
* [camel_tools](https://camel-tools.readthedocs.io/en/latest/)

### Demo

Run the tokenizer on the sample data in interactive mode.

```python greedy_disambiguator.py -t sample.in -T sample.in -o sample.out -f joint -i False -I True```

## Disambiguation Models

We have implemented 12 methods of determining the optimal tokenization for a word form given some raw, unsupervised training data.

* 1) **Ignore classes, Simple clitics** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data without conditioning over the token's class, i.e., clitic or base. We consider each clitic to be a single token. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f simple -i True```
* 2) **Consider classes, Simple clitics** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data conditioning on the token's class, i.e., clitic or base. We consider each clitic to be a single token. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f simple -i False```
* 3) **Ignore classes, Complex clitics** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data without conditioning over the token's class, i.e., clitic or base. If there are multiple proclitics, we consider the combination to be a single token, and the same goes for enclitics. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f complex -i True```
* 4) **Consider classes, Complex clitics** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data conditioning on the token's class, i.e., clitic or base. If there are multiple proclitics, we consider the combination to be a single token, and the same goes for enclitics. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f complex -i False```
* 5) **Ignore classes, Complex clitics** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data without conditioning over the token's class, i.e., clitic or base. We consider the combination of all clitics (regardless of position before or after the base) to constitute a single token. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f joint -i True```
* 6) **Consider classes, Complex clitics** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data conditioning on the token's class, i.e., clitic or base. We consider the combination of all clitics (regardless of position before or after the base) to constitute a single token. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f joint -i False```
* 7-12) **Token maximization** This is a simple baseline version of each of the first 6 models in whereby we simply choose the potential tokenization that maximizes the number of tokens. The corresponding model in 1-6 is then used only as a tie-breaker. ```... -b most_tokens```

## Usage Options

In addition to specifying the relevant disambiguation models as discussed above, you can also specify the following options at the command line:

* ```-t``` Training file. This is raw data to be run through the analyzer to gather frequency statistics and train the disambiguator.
* ```-T``` Test file. This is the data which the trained disambiguator will tokenize.
* ```-o``` Output file. This is where the tokenized output will be written.
* ```-d``` Database. This is the location of the database upon which the analyzer relies. This repository includes a built-in database which will be used by default.
* ```-c``` Cached disambiguator model. Once you train a tokenization model, you can cache it to be quickly loaded for future use. This option specifies the location where it will be cached or where a previously cached model will be looked for.
* ```-s``` Separator. This is the charactor appended to the end of prefixal elements or beginning of suffixal elements to signal how they attach to the base. The separator is '+' by default.
* ```-p``` Print options. ```most_frequent_tokens``` will print the most frequent tokens for each class to standard output. ```ranked_tokenizations_by_word``` will print the ranking over possible tokenizations for every single word in the test file.
* ```-D``` Debug mode. If True, this will print statistics for every token in every possible analyses during test time (this works in interactive mode as well).
* ```-I``` Interactive mode. If True, upon completing training, this allows the user to query the trained tokenizer from the command line.

## Acknowledgments

The Greedy Tokenizer was constructed at the New York University Abu Dhabi's [CAMeL Lab](https://nyuad.nyu.edu/en/research/centers-labs-and-projects/computational-approaches-to-modeling-language-lab.html) using the [Calima Star Analyzer](https://calimastar.abudhabi.nyu.edu/#/analyzer).
