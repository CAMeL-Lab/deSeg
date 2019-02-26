from sys import argv, stderr, stdout, exit
import os
from greedy_analyzer import Analyzer, dediacritize_normalize, replace_special_characters
from abc import ABC, abstractmethod
from scipy.stats import gmean
import pickle
import argparse
import random
import Levenshtein


GREEDY_DIR = os.path.dirname(os.path.realpath(argv[0]))

class Disambiguator_super(ABC):

    def __init__(self, analyzer, separator, ignore_class, clitic_factorization):

        self.analyzer = analyzer
        self.separator = separator
        self.ignore_class = ignore_class
        self.clitic_factorization = clitic_factorization
        self.word_2_possible_tokenizations = {}
        self.word_2_best_tokenization = {}
        assert self.clitic_factorization in ['simple', 'complex', 'joint']

        ### Determine how tokens are factorized, counted, and normalized
        self.class_2_tokens_2_frequency = {}
        self.ngrams_2_frequency = {'':0, '|':0, 'OOV':1}

        ### If we're ignoring classes, set both token classes to 'all'
        if self.ignore_class:
            self.clitic_class = 'all'
            self.base_class = 'all'
            self.class_2_tokens_2_frequency = {'all':{}}
        ### Otherwise, condition on class status, clitic vs. base
        else:
            self.clitic_class = 'clitic'
            self.base_class = 'base'
            self.class_2_tokens_2_frequency = {'clitic':{}, 'base':{}}


    def count_tokens(self, word, possible_tokenizations):
        pass  
        ### Token counting to be defined based on factorization

    
    def handle_OOV(self, word):
        pass
        ### OOV handling to be defined based on factorization
                        

    def get_possible_tokenization_statistics(self, input_file, accomodation=None):
    
        ### Get frequency of all potential tokens identified by greedy tokenizer
            ## with tokens represented with factorization granularity
            ## and by class membership as specified in command line arguments 
        for sentence in open(input_file):
            for word in sentence.split():
                word = dediacritize_normalize(word)
                word = replace_special_characters(word)

                ### Only count the possible tokenizations for unique types
                if word not in self.word_2_possible_tokenizations:
                    possible_tokenizations = self.analyzer.get_possible_tokenizations(word, accomodation=accomodation)
                    ### count tokens based on command line argument specifications
                    self.count_tokens(word, possible_tokenizations)

        ### First normalize tokens by their likelihood of being a possible analysis
            ## given that they occured as an ngram
        stderr.write('\nNormalizing token likelihoods by their likelihood to appear as an analysis given they appear as an ngram\nPerforming the normalization by the specified classes: {}..\n'.format(', '.join(list('"{}"'.format(x) for x in self.class_2_tokens_2_frequency.keys()))))
        for token_class in self.class_2_tokens_2_frequency:
            for token in self.class_2_tokens_2_frequency[token_class]:
                ngram_frequency = self.ngrams_2_frequency.get(token.replace(self.separator,''), self.ngrams_2_frequency['OOV'])
                self.class_2_tokens_2_frequency[token_class][token] /= ngram_frequency
        ### Then normalize tokens in each class by the maximum frequency in that class
        stderr.write('\nNormalizing token likelihoods by specified classes: {}..\n'.format(', '.join(list('"{}"'.format(x) for x in self.class_2_tokens_2_frequency.keys()))))
        for token_class in self.class_2_tokens_2_frequency:
            self.class_2_tokens_2_frequency[token_class] = normalize_by_maximum_frequency(self.class_2_tokens_2_frequency[token_class])


    def print_most_frequent_tokens(self):

        self.token_class_2_most_frequent_tokens = {}
        for token_class in self.class_2_tokens_2_frequency:
            stdout.write('\nMOST FREQUENT "{}" TOKENS:\n'.format(token_class))
            ranked_tokens = list(zip(self.class_2_tokens_2_frequency[token_class].values(), self.class_2_tokens_2_frequency[token_class].keys()))
            ranked_tokens.sort(reverse=True)
            self.token_class_2_most_frequent_tokens[token_class] = ranked_tokens
            for token in ranked_tokens:
                score = token[0]
                token = token[1]
                stdout.write('\n\t{}\t{}'.format(token, str(round(score, 6))))
            stdout.write('\n')
            # stdout.write('{}\n'.format('\n'.join(['\t{}\t{}'.format(x[1], str(round(x[0], 9))) for x in ranked_tokens])))


    def get_best_tokenization_per_word(self, word, baseline=None, debug=False):

        ### rank best-to-worst tokenizations for the given word
        self.word_2_best_tokenization[word] = []

        if word not in self.word_2_possible_tokenizations:
            self.handle_OOV(word)

        if baseline in ['most_tokens', 'most_tokens_no_backoff']:
            baseline_score = -1
        elif baseline in ['smallest_stem', 'smallest_stem_no_backoff']:
            baseline_score = 1000000

        ### Get scores for all possible tokenizations
        for tokenization_index in range(len(self.word_2_possible_tokenizations[word]['condition_classes'])):
            tokenization = self.word_2_possible_tokenizations[word]['condition_classes'][tokenization_index]

            scores = []
            current_tokens = 0
            current_stem_length = len(self.word_2_possible_tokenizations[word]['true_classes'][tokenization_index][1])

            ## if ignoring token classes: maximize geometric mean of all token likelihoods
            if self.ignore_class:
                for token_class in tokenization:
                    for token in tokenization[token_class]:
                        if token not in ['', '|']:
                            current_tokens += 1
                        for i in range(int(tokenization[token_class][token]+0.99)):
                            scores.append(self.class_2_tokens_2_frequency[token_class][token])

            ## if distinguishing clitics from base: maximize geometric mean of A and B
            else:
                # (A) geometric mean of clitic likelihoods
                for token in tokenization['clitic']:
                    if token not in ['', '|']:
                            current_tokens += 1
                    for i in range(int(tokenization['clitic'][token]+0.99)):
                        scores.append(self.class_2_tokens_2_frequency['clitic'][token])
                # if no clitics, just consider the base likelihood
                if len(scores) > 0:
                    scores = [gmean(scores)]

                # (B) base likelihood
                assert len(tokenization['base']) == 1
                for base in tokenization['base']:
                    current_tokens += 1
                    scores.append(self.class_2_tokens_2_frequency[self.base_class][base])

            ### Record score
            score = gmean(scores)
            if debug:
                stdout.write('{}\n\t{}\n\t{}\n\t{}\n'.format(word, str(tokenization), ', '.join(str(round(x, 5)) for x in scores), str(round(score, 5))))
            tokenization = self.word_2_possible_tokenizations[word]['true_classes'][tokenization_index]

            ### Handle most_tokens baselines
            if baseline in ['most_tokens', 'most_tokens_no_backoff']:
                if current_tokens > baseline_score:
                    baseline_score = current_tokens
                    self.word_2_best_tokenization[word] = [[score, tokenization]]
                elif current_tokens == baseline_score:
                    self.word_2_best_tokenization[word].append([score, tokenization])

            ### Handle smallest_stem baselines
            elif baseline in ['smallest_stem', 'smallest_stem_no_backoff']:
                if current_stem_length < baseline_score:
                    baseline_score = current_stem_length
                    self.word_2_best_tokenization[word] = [[score, tokenization]]
                elif current_stem_length == baseline_score:
                    self.word_2_best_tokenization[word].append([score, tokenization])

            else:
                self.word_2_best_tokenization[word].append([score, tokenization])

        if len(self.word_2_best_tokenization[word]) == 0:
            stderr.write('NO TOKENIZATIONS FOUND!!!{}\n'.format('\n\t'.join(str(x) for x in [word, word_2_possible_tokenizations[word]])))
            exit()

        else:
            if baseline in ['most_tokens_no_backoff', 'smallest_stem_no_backoff']:
                random.shuffle(self.word_2_best_tokenization[word])
            else:
                self.word_2_best_tokenization[word].sort(reverse=True)

        # print(word)
        # print('FINAL RANKING:\n\t{}'.format('\n\t'.join(list('{}\n\t\t{}'.format(str(x[1]), str(x[0])) for x in self.word_2_best_tokenization[word]))))


    def print_ranked_tokenizations_by_word(self):

        for word in self.word_2_best_tokenization:
            ranked_tokenizations = self.word_2_best_tokenization[word]
            stdout.write('\nBEST TOKENIZATIONS FOR "{}":\n'.format(word))
            for tokenization in ranked_tokenizations:
                score = tokenization[0]
                tokens = list(tokenization[1][0])
                tokens.append(tokenization[1][1])
                tokens.extend(tokenization[1][2]) 
                stdout.write('\n\t{}\t{}'.format(' '.join(tokens), str(round(score, 30))))
            stdout.write('\n')
            # stdout.write('{}\n'.format('\n'.join(['\t{}\t{}'.format(' '.join([' '.join(x[1][0]), x[1], ' '.join(x[1][2])]), str(round(x[0], 3))) for x in self.word_2_best_tokenization[word]])))


    def interact(self, debug=False, baseline=None):

        os.system('clear')
        stdout.write('Welcome to interactive mode!\nYou can enter q at any time to quit.\n\nSo like what do you wanna tokenize or whatever?\n\n: ')

        while True:

            sentence = input()
            
            if sentence in ['q', 'Q']:
                exit()

            stdout.write('{}\n\n\n\n: '.format(' '.join(self.tokenize_sentence(sentence, debug=debug, baseline=baseline))))


    def tokenize_sentence(self, sentence, baseline=None, debug=False):

        tokenized_sentence = []
        for word in sentence.split():
            word = dediacritize_normalize(word)
            word = replace_special_characters(word)
            if word not in self.word_2_best_tokenization:
                self.get_best_tokenization_per_word(word, baseline=baseline, debug=debug)
            tokenization = self.word_2_best_tokenization[word][0][1]
            for proclitic in tokenization[0]:
                tokenized_sentence.append(proclitic)
            tokenized_sentence.append(tokenization[1])
            for enclitic in tokenization[2]:
                tokenized_sentence.append(enclitic)

        return tokenized_sentence


    def apply_tokenization(self, input_file, output_file, baseline=None, debug=False):
        ### Go through corpus again and assign the optimal tokenization to each word
        output = open(output_file, 'w')
        for sentence in open(input_file):
            output.write('{}\n'.format(' '.join(self.tokenize_sentence(sentence, baseline=baseline))))
        output.close()


class Disambiguator_simpleFactorization(Disambiguator_super):

    ### Define how tokens are counted with simple facorization
    def count_tokens(self, word, possible_tokenizations):
        
        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            cat_tok = '{}{}{}'.format(''.join(tokenization[0]), tokenization[1], ''.join(tokenization[2])).replace(self.separator, '')
            discounted_increment = 1 / (Levenshtein.distance(word, cat_tok) + 1)

            base = tokenization[1]
            ### register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:discounted_increment}})

            ### prepare to register clitics
            if self.clitic_class != self.base_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}

            ## check if proclitic is blank
            if len(tokenization[0]) == 0:
                if '' not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][''] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][''] += discounted_increment

            ## record proclitics
            for proclitic in tokenization[0]:

                if proclitic not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] += discounted_increment

            ## check if enclitic is blank
            if len(tokenization[2]) == 0:
                if '' not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][''] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][''] += discounted_increment

            ## record enclitics
            for enclitic in tokenization[2]:

                if enclitic not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] += discounted_increment

        ### Update total token frequency counts given the word's possible tokens
        for possible_tokenization in self.word_2_possible_tokenizations[word]['condition_classes']:
            for token_class in self.class_2_tokens_2_frequency:
                for token in possible_tokenization[token_class]:

                    if token not in self.class_2_tokens_2_frequency[token_class]:
                        self.class_2_tokens_2_frequency[token_class][token] = 0
                    self.class_2_tokens_2_frequency[token_class][token] += possible_tokenization[token_class][token]

        ### Update ngram frequency counts
        len_word = len(word)
        len_word_1 = len_word + 1
        self.ngrams_2_frequency[''] += len_word_1
        for start in range(len_word):
            for finish in range(start+1, len_word_1):
                ngram = word[start:finish]
                if ngram not in self.ngrams_2_frequency:
                    self.ngrams_2_frequency[ngram] = 0
                self.ngrams_2_frequency[ngram] += 1

    ### And how OOVs are handled with simple factorization
    def handle_OOV(self, word):

        possible_tokenizations = self.analyzer.get_possible_tokenizations(word)

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            cat_tok = '{}{}{}'.format(''.join(tokenization[0]), tokenization[1], ''.join(tokenization[2])).replace(self.separator, '')
            discounted_increment = 1 / (Levenshtein.distance(word, cat_tok) + 1)

            base = tokenization[1]
            if base not in self.class_2_tokens_2_frequency[self.base_class]:
                base = 'OOV'

            ### register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:discounted_increment}})

            ### prepare to register clitics
            if self.clitic_class != self.base_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}

            if len(tokenization[0]) == 0:
                if '' not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][''] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][''] += discounted_increment

            for proclitic in tokenization[0]:

                if proclitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                    proclitic = 'OOV'

                if proclitic not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] += discounted_increment

            if len(tokenization[2]) == 0:
                if '' not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][''] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][''] += discounted_increment

            for enclitic in tokenization[2]:

                if enclitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                    enclitic = 'OOV'

                if enclitic not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] += discounted_increment


class Disambiguator_complexFactorization(Disambiguator_super):

    ### Define how tokens are counted with complex facorization
    def count_tokens(self, word, possible_tokenizations):

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:



            proclitic = ''.join(tokenization[0])
            enclitic = ''.join(tokenization[2])
            base = tokenization[1]

            cat_tok = '{}{}{}'.format(proclitic, base, enclitic).replace(self.separator, '')
            discounted_increment = 1 / (Levenshtein.distance(word, cat_tok) + 1)

            # register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:discounted_increment}})

            # register complex clitics that exist
            if self.base_class != self.clitic_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}
            for clitic in [proclitic, enclitic]:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][clitic] = discounted_increment

        ### Update total token frequency counts given the word's possible tokens
        for possible_tokenization in self.word_2_possible_tokenizations[word]['condition_classes']:
            for token_class in self.class_2_tokens_2_frequency:
                for token in possible_tokenization[token_class]:
                    if token not in self.class_2_tokens_2_frequency[token_class]:
                        self.class_2_tokens_2_frequency[token_class][token] = 0
                    self.class_2_tokens_2_frequency[token_class][token] += possible_tokenization[token_class][token]

        ### Update ngram frequency counts
        len_word = len(word)
        len_word_1 = len_word + 1
        self.ngrams_2_frequency[''] += len_word_1
        for start in range(len_word):
            for finish in range(start+1, len_word_1):
                ngram = word[start:finish]
                if ngram not in self.ngrams_2_frequency:
                    self.ngrams_2_frequency[ngram] = 0
                self.ngrams_2_frequency[ngram] += 1

    ### Define how OOVs are handled with complex facorization
    def handle_OOV(self, word):

        possible_tokenizations = self.analyzer.get_possible_tokenizations(word)

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            proclitic = ''.join(tokenization[0])
            enclitic = ''.join(tokenization[2])
            base = tokenization[1]

            cat_tok = '{}{}{}'.format(proclitic, base, enclitic).replace(self.separator, '')
            discounted_increment = 1 / (Levenshtein.distance(word, cat_tok) + 1)

            if proclitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                proclitic = 'OOV'
            if enclitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                enclitic = 'OOV'
            if base not in self.class_2_tokens_2_frequency[self.base_class]:
                base = 'OOV'

            # register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:discounted_increment}})

            # register complex clitics that exist
            if self.base_class != self.clitic_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}
            self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] = discounted_increment
            self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] = discounted_increment


class Disambiguator_jointFactorization(Disambiguator_super):

    ### Define how tokens are counted with joint facorization
    def count_tokens(self, word, possible_tokenizations):

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            clitic = '{}|{}'.format(''.join(tokenization[0]), ''.join(tokenization[2]))
            base = tokenization[1]

            cat_tok = '{}{}{}'.format(''.join(tokenization[0]), base, ''.join(tokenization[2])).replace(self.separator, '')
            discounted_increment = 1 / (Levenshtein.distance(word, cat_tok) + 1)

            # register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:discounted_increment}})

            # register circumclitic if it exists
            if self.base_class != self.clitic_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}
            self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][clitic] = discounted_increment

        ### Update total token frequency counts given the word's possible tokens
        for possible_tokenization in self.word_2_possible_tokenizations[word]['condition_classes']:
            for token_class in self.class_2_tokens_2_frequency:
                for token in possible_tokenization[token_class]:
                    if token not in self.class_2_tokens_2_frequency[token_class]:
                        self.class_2_tokens_2_frequency[token_class][token] = 0
                    self.class_2_tokens_2_frequency[token_class][token] += possible_tokenization[token_class][token]

        ### Update ngram frequency counts
        len_word = len(word)
        len_word_1 = len_word + 1
        self.ngrams_2_frequency[''] += len_word_1
        self.ngrams_2_frequency['|'] += len_word_1 * ( len_word/2 )
        for start in range(len_word):
            for finish in range(start+1, len_word_1):
                ngram = word[start:finish]

                ### count joint ngrams containing both a proclitic and enclitic
                joint_ngram = '{}|'.format(ngram)
                for start2 in range(finish, len_word):
                    for finish2 in range(start2+1, len_word_1):
                        joint_ngram = ''.join([joint_ngram,word[start2:finish2]])
                        if joint_ngram not in self.ngrams_2_frequency:
                            self.ngrams_2_frequency[joint_ngram] = 0
                        self.ngrams_2_frequency[joint_ngram] += 1

                ### count normal ngrams
                if ngram not in self.ngrams_2_frequency:
                    self.ngrams_2_frequency[ngram] = 0
                self.ngrams_2_frequency[ngram] += 1

                ### count joint ngrams where there is no enclitic
                joint_ngram = '{}|'.format(ngram)
                if joint_ngram not in self.ngrams_2_frequency:
                    self.ngrams_2_frequency[joint_ngram] = 0
                self.ngrams_2_frequency[joint_ngram] += 1

                ### count joint ngrams where there is no proclitic
                joint_ngram = '|{}'.format(ngram)
                if joint_ngram not in self.ngrams_2_frequency:
                    self.ngrams_2_frequency[joint_ngram] = 0
                self.ngrams_2_frequency[joint_ngram] += 1

    ### Define how OOVs are handled with joint facorization
    def handle_OOV(self, word):

        possible_tokenizations = self.analyzer.get_possible_tokenizations(word)

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            clitic = '{}|{}'.format(''.join(tokenization[0]), ''.join(tokenization[2]))
            if clitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                clitic = 'OOV'
            base = tokenization[1]
            cat_tok = '{}{}{}'.format(''.join(tokenization[0]), base, ''.join(tokenization[2])).replace(self.separator, '')
            discounted_increment = 1 / (Levenshtein.distance(word, cat_tok) + 1)
            if base not in self.class_2_tokens_2_frequency[self.base_class]:
                base = 'OOV'

            # register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:discounted_increment}})

            # register circumclitic if it exists
            if self.base_class != self.clitic_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}
            self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][clitic] = discounted_increment


def normalize_by_maximum_frequency(dictionary):

    values = list(dictionary.values())
    max_frequency = max(values)
    smoothing_minimum = min(values)/max_frequency
    for key in dictionary:
        dictionary[key] /= max_frequency
    dictionary['OOV'] = smoothing_minimum

    return dictionary


def str2bool(v):

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_disambiguator_filename(args):

    disambiguator_file = args.cached_disambiguator
    if disambiguator_file == None:
        disambiguator_file = 'disambiguator_{}_{}_{}_minBase{}'.format(os.path.basename(args.train), os.path.basename(args.database), args.clitic_factorization, str(args.min_base_length))
        if args.ignore_class:
            disambiguator_file += '_unconditional.pkl'
        else:
            disambiguator_file += '_classConditional.pkl'
        disambiguator_file = os.path.join(GREEDY_DIR, disambiguator_file)

    return disambiguator_file


def pickleIn(file):

    pklFile = open(file, 'rb')
    ofTheKing =pickle.load(pklFile)
    pklFile.close()

    return ofTheKing


def pickleOut(thingy, file):

    pklFile = open(file, 'wb')
    pickle.dump(thingy, pklFile)
    pklFile.close()


#########################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'apply', 'interactive'], help='You can run the Greedy Tokenizer in either "train" or "apply" mode', required=True)
    parser.add_argument('-t', '--train', type=str, help='Location of the corpus from which we will learn the maximum likelihood greedy tokenization scheme', required=False, default=None)
    parser.add_argument('-T', '--test', type=str, help='Location of the corpus to which we will apply the learned maximum likelihood greedy tokenization scheme', required=False, default=None)
    parser.add_argument('-o', '--output', type=str, help='Location to which the tokenized corpus will be written out', required=False, default='output.tok')
    parser.add_argument('-d', '--database', type=str, help='Database to be used by the analyzer', required=False, default='built-in')
    parser.add_argument('-a', '--accomodation', type=str, choices=['built-in', 'DA'], help='Triggers ad hoc accomodation available to the analyzer necessary to reformat certain databases outputs into the expected format', required=False, default=None)
    parser.add_argument('-c', '--cached_disambiguator', type=str, help='Where to load or store the trained disambiguator', required=False, default=None)
    parser.add_argument('-s', '--separator', type=str, help='Separator used to mark clitic boundaries', required=False, default='+')
    parser.add_argument('-l', '--min_base_length', type=int, help='Minimum length of the base word after separating clitics for the analysis to be considered feasible.', required=False, default=1)
    parser.add_argument('-i', '--ignore_class', type=str2bool, help="If True, optimal tokenizations are chosen based on geometric mean likelihood all proposed component tokens. Otherwise, token likelihoods are calculated conditional on class, i.e., clitic vs. base, and normalized by the most likely member of their class. Then, optimal tokenizations are chosen based on the geometric mean of A and B; where A is the geometric mean of component clitic likelihoods and B is the base likelihood. When no clitics are proposed, the tokenization's score is simply the base likelihood.", required=False, default=False)
    parser.add_argument('-f', '--clitic_factorization', type=str, choices=['simple','complex','joint'], help="When computing likelihood of tokenization components, we can either consider the likelihood of each clitic token independently (simple), or we can consider the joint likelihood of the entire proclitic and the joint likelihood of the entire enclitic (complex), or we can consider the joint likelihood of the entire exponence, i.e., the cicumfix consisting of proclitic + enclitic (joint).", required=False, default='joint')
    parser.add_argument('-b', '--baseline', type=str, choices=['most_tokens', 'smallest_stem', 'most_tokens_no_backoff', 'smallest_stem_no_backoff'], help="Baseline models that primarily maximize the number of tokens or minimize the length of the base and either secondarily maximize likelihood as a tie breaker or randomly pick a tie breaker which is used consistently for every instance of the type in question.", required=False, default=None)
    parser.add_argument('-p', '--print_options', nargs='+', help="Optional print statements that can be executed for debugging purposes. They report the most (token) frequent proclictics, enclitics, and bases chosen by the disambiguator and/or a ranking for each word of the disambiguator's preferences over the analyzer's proposed tokenizations.", required=False, choices=['most_frequent_tokens', 'ranked_tokenizations_by_word'], default=[])
    parser.add_argument('-D', '--debug', type=str2bool, help="Compute tokenizations in debug mode.", required=False, default=False)


    args = parser.parse_args()    


    ### TRAINING MODE
    if args.mode == 'train':

        # Initialize the analyzer
        stderr.write('\nInitializing analyzer with database "{}"..\n'.format(args.database))
        analyzer = Analyzer(args.database, args.separator, args.min_base_length)

        if analyzer.database_file == 'built-in':
            args.accomodation = args.database

        # Train the disambiguator
        disambiguator_file = get_disambiguator_filename(args)
        try:
            disambiguator = pickleIn(disambiguator_file)
            command = 'python greedy_disambiguator.py -m apply -c {} -T [data_to_tokenize] -o [desired_output_file]'.format(disambiguator_file)
            stderr.write('\nDisambiguator "{}" has already been trained!\nTo apply the disambiguator, run the following command:\n{}\n'.format(disambiguator_file, command))
            exit()
        except FileNotFoundError:
            stderr.write('\nTraining disambiguator on "{}"..\n'.format(args.train))
            if args.clitic_factorization == 'simple':
                disambiguator = Disambiguator_simpleFactorization(analyzer, args.separator, args.ignore_class, args.clitic_factorization)
            elif args.clitic_factorization == 'complex':
                disambiguator = Disambiguator_complexFactorization(analyzer, args.separator, args.ignore_class, args.clitic_factorization)
            elif args.clitic_factorization == 'joint':
                disambiguator = Disambiguator_jointFactorization(analyzer, args.separator, args.ignore_class, args.clitic_factorization)
            disambiguator.get_possible_tokenization_statistics(args.train, accomodation=args.accomodation)

        # Save the trained disambiguator    
        stderr.write('\nCaching trained disambiguator..\n')
        pickleOut(disambiguator, disambiguator_file)

        # Move on to bigger and better things
        if len(args.print_options) > 0:
            stderr.write('\nThe print_options can only be used in Apply mode. You can apply the trained disambiguator to "{}" though to print any relevant statistics from the training set.'.format(args.train))
        stderr.write('\nDone! _\n     / \\ \n    /   \\    \n __/_____\\__\n   /     \\\n  | \\   / |\n| |   v   | |\n| |  ___  | |\n|  \\_____/  |\n \\____|____/\n      |\n      |\n      |\n      |\n      |\n     / \\\n    /   \\\
        \n   /     \\\n  |       |\n  |       |\n  |       |\nSo yeah, your trained tokenizer is here: "{}"\n\n'.format(disambiguator_file))



    # Read in the trained disambiguator
    else:
        try:
            stderr.write('\nReading in the pre-trained disambiguator "{}"\n'.format(args.cached_disambiguator))
            disambiguator = pickleIn(args.cached_disambiguator)
        except FileNotFoundError:
            stderr.write('\tDisambiguator not found!!!\n'.format(disambiguator_file))
            exit()



    ### INTERACTIVE MODE
    if args.mode == 'interactive':

        disambiguator.interact(debug=args.debug, baseline=args.baseline)



    ### APPLY TOKENIZATION MODE
    elif args.mode == 'apply':

        # Apply the trained disambiguator and write out tokenization
        stderr.write('\nApplying maximum likelihood greedy tokenization..\n\tReading input from "{}" and writing output to: "{}"..\n'.format(args.test, args.output))
        disambiguator.apply_tokenization(args.test, args.output, baseline=args.baseline, debug=args.debug)

        # Print out any requested information
        if 'most_frequent_tokens' in args.print_options:
            disambiguator.print_most_frequent_tokens()
        if 'ranked_tokenizations_by_word' in args.print_options:
            disambiguator.print_ranked_tokenizations_by_word()
        if len(args.print_options) == 0:
            stderr.write('\n')
            os.system('clear')

        # Shake your tailfeather
        stderr.write('Done! _\n     / \\ \n    /   \\    \n __/_____\\__\n   /     \\\n  | o   o |\n| |   v   | |\n| | \\___/ | |\n|  \\_____/  |\n \\____|____/\n      |\n      |\n      |\n      |\n      |\n     / \\\n    /   \\\
        \n   /     \\\n  |       |\n  |       |\n  |       |\nSo yeah, your tokenized output is here: "{}"\n\n'.format(args.output))
