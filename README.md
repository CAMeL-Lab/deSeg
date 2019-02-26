# Greedy Tokenizer

Given an analyzer class with a method that returns all possible tokenizations of an input word, we train a greedy, unsupervised tokenization model to disambiguate these possible tokenizations. Tokenization depends only on statistics of observed possible tokenizations during training, i.e., context is not considered in this implementation. Thus, all word types will be deterministically tokenized regardless of context. The quality of the tokenization is completely dependent on the assumption that tokens frequently proposed by the analyzer are more likely to appear in the correct tokenization than tokens which are proposed less frequently.

The analyzer included in this repository runs on Modern Standard Arabic (which is the language of *sample.in*, the white space/punctuation separated sample training data). Our goal was to tokenize all clitics, but you can easily replace the analyzer and use the disambiguator to tokenize other pre/suffixal morphemes. The disambiguator just expects the *get_possible_tokenizations()* method of the analyzer to return a list of triples for each input word, where each triple is an analysis, comprised of a potentially empty list of prefixal elements, the base string, and a potentially empty list of suffixal elements.

We intend for this implementation to be adapted to handle under resourced languages where an analyzer of closed class morphemes can be cheaply built, but there is no lexical information available. That is to say that we have or can quickly build a database of the possible affixes and associated morpho-syntactic behavior, but we don't have any information about the existance or combinatorics of open class stems. An analyzer built on such a database would produce the set of analyses that includes any allowable combination of prefixes and suffixes that could be realized by a given word, but it would not be able to prune any of these as invalid because it would not be able to check if the remaining stem is a valid open class stem or not.

## Quick Start

This is a short demo. For a full description of the various models you can train with the Greedy Tokenizer, see Disambiguation Models below.

### Prerequisites

* [Python 3](https://www.python.org/downloads/)
* [Scipy](https://www.scipy.org)
* [CAMeL Tools](https://camel-tools.readthedocs.io/en/latest/)

### Demo

Train (unsupervised) the tokenizer on punctuation-separated input text (*sample.in*)

```python greedy_disambiguator.py -m train -t sample.in -d built-in -f simple -i False -l 2```

Run the cached trained tokenizer from above in interactive mode

```python greedy_disambiguator.py -m interactive -c disambiguator_sample.in_built-in_simple_minBase2_classConditional.pkl```

Apply the cached trained tokenizer to a test file (*sample.in*) and generate a tokenized output (*sample.out*)

```python greedy_disambiguator.py -m apply -c disambiguator_sample.in_built-in_simple_classConditional.pkl -T sample.in -o sample.out```

## Training Disambiguation Models

The Greedy Tokenizer is run in train mode with the following option: ```-m train```. It can train any of 5 models as follows by manipulating the ```f```, clitic factorization, and ```i```, ignore classes options:

1. **SU: Simple clitics, Unconditional** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data without conditioning over the token's class, i.e., clitic or base. Likelihood is calculated as the frequency with which the token appeared in an analysis divided by the number of times the token appeared as an ngram (if there are re-write rules involved, we discount appearances based on edit distance between the input form and the re-written form). We consider each clitic to be a single token. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f simple -i True```
2. **SC: Simple clitics, Class conditional** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data conditioning on the token's class, i.e., clitic or base. We consider each clitic to be a single token. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f simple -i False```
3. **CU: Complex clitics, Unconditional** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data without conditioning over the token's class, i.e., clitic or base. If there are multiple proclitics, we consider the combination to be a single token, and the same goes for enclitics. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f complex -i True```
4. **CC: Complex clitics, Class conditional** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data conditioning on the token's class, i.e., clitic or base. If there are multiple proclitics, we consider the combination to be a single token, and the same goes for enclitics. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. ```... -f complex -i False```
5. **J: Joint clitics** We calculate the likelihood for each token (clitic or base) over all occurences in all possible analyses in the training data without conditioning over the token's class, i.e., clitic or base. We consider the combination of all clitics (regardless of position before or after the base) to constitute a single token. The optimal tokenization for a given word is the possible analysis which maximizes the geometric mean of its component tokens. Mathematically, the unconditional and conditional variants are identical with joint clitics because there can only ever be a single clitic per analyses in the joint model, so it does not matter if classes are ignored or not. ```... -f joint -i True/False```

## Applying Disambiguation Models

To apply a trained disambiguator model, you need to specify apply mode with the option ```-m apply```. You will also need to specify the cached trained disambiguator model you wish to use. A trained disambiguator is always stored in the working directory as a pickle file at the conclusion of training and the location of the file is written to standard output. You can specify which cached disambiguation model to apply during testing with the cached model option: ```-c [your_disambiguator.pkl]```

## Usage Options

In addition to specifying the relevant disambiguation models as discussed above, you can also specify the following options at the command line:

* ```-t``` Training file. This is raw data to be run through the analyzer to gather frequency statistics and train the disambiguator.
* ```-T``` Test file. This is the data which the trained disambiguator will tokenize.
* ```-o``` Output file. This is where the tokenized output will be written.
* ```-d``` Database. This is the location of the database upon which the analyzer relies. This repository includes a built-in database which will be used by default.
* ```-a``` Accomodation. This argument is used to trigger any ad hoc function you need to include in order to make your database behave in the manner that the analyzer expects. Included accomodations are ```DA``` which is designed to handle a delexicalized Dialectal Arabic database and ```built-in``` which will be triggered by default if you use the default, built-in Modern Standard Arabic database.
* ```-l``` Minimum base length. The default is 1 character. If no analysis can be found with base of length >= the minimum base length, the word will be returned as the base token and it will be assumed that there are no clitics.
* ```-b``` Baseline mode. There are four baseline options. ```most_tokens``` chooses the analysis that maximizes the number of tokens for each word form, using the trained model only to break ties. ```smallest_stem``` chooses the analysis that minimizes the stem length such that it remains >= the minimum length specified by option ```-l```, and, again, the trained model is only used to break ties. ```most_tokens_no_backoff``` and ```smallest_stem_no_backoff``` are the same as their counterparts except, for each type, ties are broken by random chance. Once a random choice of best tokenization is made for each word though, that tokenization is used consistently over every occurence of said word.
* ```-c``` Cached disambiguator model. Once you train a tokenization model, you can cache it to be quickly loaded for future use. This option specifies the location where it will be cached or where a previously cached model will be looked for.
* ```-s``` Separator. This is the charactor appended to the end of prefixal elements or beginning of suffixal elements to signal how they attach to the base. The separator is '+' by default.
* ```-p``` Print options. ```most_frequent_tokens``` will print the most frequent tokens for each class to standard output. ```ranked_tokenizations_by_word``` will print the ranking over possible tokenizations for every single word in the test file.
* ```-D``` Debug mode. If True, this will print statistics for every token in every possible analyses during test time (this works in interactive mode as well).
* ```-m``` In addition to ```train``` and ```apply```, the greedy tokenizer can also be run in interactive mode by using the ```interactive``` option. This allows the user to query the trained tokenizer from the command line.

## Acknowledgments

The Greedy Tokenizer was constructed at the New York University Abu Dhabi's [CAMeL Lab](https://nyuad.nyu.edu/en/research/centers-labs-and-projects/computational-approaches-to-modeling-language-lab.html) using the [Calima Star Analyzer](https://calimastar.abudhabi.nyu.edu/#/analyzer).
