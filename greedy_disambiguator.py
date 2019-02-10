from sys import argv, stderr, stdout
import os
from greedy_analyzer import Analyzer, dediacritize_normalize, replace_special_characters
from abc import ABC, abstractmethod
from scipy.stats import gmean
import pickle
import argparse


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
                        

    def get_possible_tokenization_statistics(self, input_file):
    
        ### Get frequency of all potential tokens identified by greedy tokenizer
            ## with tokens represented with factorization granularity
            ## and by class membership as specified in command line arguments 
        for sentence in open(input_file):
            for word in sentence.split():
                word = dediacritize_normalize(word)
                word = replace_special_characters(word)

                ### Only get the possible tokenizations the first time you see each word
                if word not in self.word_2_possible_tokenizations:
                    possible_tokenizations = self.analyzer.get_possible_tokenizations(word)
                    ### count tokens based on command line argument specifications
                    self.count_tokens(word, possible_tokenizations)

                ### Update total token frequency counts given the word's possible tokens
                for possible_tokenization in self.word_2_possible_tokenizations[word]['condition_classes']:
                    for token_class in self.class_2_tokens_2_frequency:
                        for token in possible_tokenization[token_class]:
                            if token not in self.class_2_tokens_2_frequency[token_class]:
                                self.class_2_tokens_2_frequency[token_class][token] = 0
                            self.class_2_tokens_2_frequency[token_class][token] += possible_tokenization[token_class][token]

        ### Normalize tokens in each class by the maximum frequency in that class
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

        ### Get scores for all possible tokenizations
        for tokenization_index in range(len(self.word_2_possible_tokenizations[word]['condition_classes'])):
            tokenization = self.word_2_possible_tokenizations[word]['condition_classes'][tokenization_index]
            scores = []
            score_base = 0

            ## if you're just maximizing number of tokens: count them up
            if baseline == 'most_tokens':
                for token_class in tokenization:
                    for token in tokenization[token_class]:
                        score_base += 1000*tokenization[token_class][token]

            ## if ignoring token classes: maximize geometric mean of all token likelihoods
            if self.ignore_class:
                for token_class in tokenization:
                    for token in tokenization[token_class]:
                        for i in range(tokenization[token_class][token]):
                            scores.append(score_base + self.class_2_tokens_2_frequency[token_class][token])

            ## if distinguishing clitics from base: maximize geometric mean of A and B
            else:
                # (A) geometric mean of clitic likelihoods
                for token in tokenization['clitic']:
                    for i in range(tokenization['clitic'][token]):
                        scores.append(score_base + self.class_2_tokens_2_frequency['clitic'][token])
                # if no clitics, just consider the base likelihood
                if len(scores) > 0:
                    scores = [gmean(scores)]

                # (B) base likelihood
                assert len(tokenization['base']) == 1
                for base in tokenization['base']:
                    scores.append(score_base + self.class_2_tokens_2_frequency[self.base_class][base])

            ### Record score
            score = gmean(scores)
            if debug:
                stdout.write('{}\n\t{}\n\t{}\n\t{}\n'.format(word, str(tokenization), ', '.join(str(round(x, 5)) for x in scores), str(round(score, 5))))
            tokenization = self.word_2_possible_tokenizations[word]['true_classes'][tokenization_index]

            self.word_2_best_tokenization[word].append([score, tokenization])

        if len(self.word_2_best_tokenization[word]) == 0:
            stderr.write('NO TOKENIZATIONS FOUND!!!{}\n'.format('\n\t'.join(str(x) for x in [word, word_2_possible_tokenizations[word]])))
            exit()

        else:
            self.word_2_best_tokenization[word].sort(reverse=True)


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


    def interact(self, debug=False):

        os.system('clear')
        stdout.write('Welcome to interactive mode!\nYou can enter q to quit or c to continue at any time.\n\nSo like what do you wanna tokenize or whatever?\n\n: ')

        while True:

            sentence = input()
            
            if sentence in ['q', 'Q']:
                exit()
            if sentence in ['c', 'C']:
                break
            stdout.write('{}\n\n\n\n: '.format(' '.join(self.tokenize_sentence(sentence, debug=debug))))


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
            output.write('{}\n'.format(' '.join(self.tokenize_sentence(sentence))))
        output.close()


class Disambiguator_simpleFactorization(Disambiguator_super):

    ### Define how tokens are counted with simple facorization
    def count_tokens(self, word, possible_tokenizations):
        
        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            base = tokenization[1]
            ### register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:1}})

            ### prepare to register clitics
            if self.clitic_class != self.base_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}

            for proclitic in tokenization[0]:

                if proclitic not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] += 1

            for enclitic in tokenization[2]:

                if enclitic not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] += 1

    ### And how OOVs are handled with simple factorization
    def handle_OOV(self, word):

        possible_tokenizations = self.analyzer.get_possible_tokenizations(word)

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            base = tokenization[1]
            if base not in self.class_2_tokens_2_frequency[self.base_class]:
                base = 'OOV'

            ### register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:1}})

            ### prepare to register clitics
            if self.clitic_class != self.base_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}

            for proclitic in tokenization[0]:

                if proclitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                    proclitic = 'OOV'

                if proclitic not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] += 1

            for enclitic in tokenization[2]:

                if enclitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                    enclitic = 'OOV'

                if enclitic not in self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class]:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] = 0
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] += 1


class Disambiguator_complexFactorization(Disambiguator_super):

    ### Define how tokens are counted with complex facorization
    def count_tokens(self, word, possible_tokenizations):

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            proclitic = ''.join(tokenization[0])
            enclitic = ''.join(tokenization[2])
            base = tokenization[1]

            # register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:1}})

            # register complex clitics that exist
            if self.base_class != self.clitic_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}
            for clitic in [proclitic, enclitic]:
                if len(clitic) > 0:
                    self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][clitic] = 1

    ### Define how OOVs are handled with complex facorization
    def handle_OOV(self, word):

        possible_tokenizations = self.analyzer.get_possible_tokenizations(word)

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            pro = True
            proclitic = ''.join(tokenization[0])
            if len(proclitic) == 0:
                pro = False
            elif proclitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                proclitic = 'OOV'
            en = True
            enclitic = ''.join(tokenization[2])
            if len(enclitic) == 0:
                en = False
            elif enclitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                enclitic = 'OOV'
            base = tokenization[1]
            if base not in self.class_2_tokens_2_frequency[self.base_class]:
                base = 'OOV'

            # register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:1}})

            # register complex clitics that exist
            if self.base_class != self.clitic_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}
            if pro:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][proclitic] = 1
            if en:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][enclitic] = 1


class Disambiguator_jointFactorization(Disambiguator_super):

    ### Define how tokens are counted with joint facorization
    def count_tokens(self, word, possible_tokenizations):

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            clitic = '{}|{}'.format(''.join(tokenization[0]), ''.join(tokenization[2]))
            base = tokenization[1]

            # register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:1}})

            # register circumclitic if it exists
            if self.base_class != self.clitic_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}
            if len(clitic) > 1:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][clitic] = 1

    ### Define how OOVs are handled with joint facorization
    def handle_OOV(self, word):

        possible_tokenizations = self.analyzer.get_possible_tokenizations(word)

        self.word_2_possible_tokenizations[word] = {'true_classes':possible_tokenizations, 'condition_classes':[]}

        for tokenization in possible_tokenizations:

            cl = True
            clitic = '{}|{}'.format(''.join(tokenization[0]), ''.join(tokenization[2]))
            if len(clitic) == 1:
                cl = False
            elif clitic not in self.class_2_tokens_2_frequency[self.clitic_class]:
                clitic = 'OOV'
            base = tokenization[1]
            if base not in self.class_2_tokens_2_frequency[self.base_class]:
                base = 'OOV'

            # register base
            self.word_2_possible_tokenizations[word]['condition_classes'].append({self.base_class:{base:1}})

            # register circumclitic if it exists
            if self.base_class != self.clitic_class:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class] = {}
            if cl:
                self.word_2_possible_tokenizations[word]['condition_classes'][-1][self.clitic_class][clitic] = 1


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
        disambiguator_file = 'disambiguator_{}_{}_{}'.format(os.path.basename(args.train), os.path.basename(args.database), args.clitic_factorization)
        if args.ignore_class:
            disambiguator_file += '_classless.pkl'
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
    parser.add_argument('-t', '--train', type=str, help='Location of the corpus from which we will learn the maximum likelihood greedy tokenization scheme', required=False, default=None)
    parser.add_argument('-T', '--test', type=str, help='Location of the corpus to which we will apply the learned maximum likelihood greedy tokenization scheme', required=False, default=None)
    parser.add_argument('-o', '--output', type=str, help='Location to which the tokenized corpus will be written out', required=False, default='output.tok')
    parser.add_argument('-d', '--database', type=str, help='Database to be used by the analyzer', required=False, default='built-in')
    parser.add_argument('-c', '--cached_disambiguator', type=str, help='Where to load or store the trained disambiguator', required=False, default=None)
    parser.add_argument('-s', '--separator', type=str, help='Separator used to mark clitic boundaries', required=False, default='+')
    parser.add_argument('-i', '--ignore_class', type=str2bool, help="If True, optimal tokenizations are chosen based on geometric mean likelihood all proposed component tokens. Otherwise, token likelihoods are calculated conditional on class, i.e., clitic vs. base, and normalized by the most likely member of their class. Then, optimal tokenizations are chosen based on the geometric mean of A and B; where A is the geometric mean of component clitic likelihoods and B is the base likelihood. When no clitics are proposed, the tokenization's score is simply the base likelihood.", required=False, default=False)
    parser.add_argument('-f', '--clitic_factorization', type=str, choices=['simple','complex','joint'], help="When computing likelihood of tokenization components, we can either consider the likelihood of each clitic token independently (simple), or we can consider the joint likelihood of the entire proclitic and the joint likelihood of the entire enclitic (complex), or we can consider the joint likelihood of the entire exponence, i.e., the cicumfix consisting of proclitic + enclitic (joint).", required=False, default='joint')
    parser.add_argument('-b', '--baseline', type=str, choices=['most_tokens'], help="Baseline model that primarily maximizes the number of tokens and secondarily maximizes likelihood.", required=False, default=None)
    parser.add_argument('-p', '--print_options', nargs='+', help="Optional print statements that can be executed for debugging purposes. They report the most (token) frequent proclictics, enclitics, and bases chosen by the disambiguator and/or a ranking for each word of the disambiguator's preferences over the analyzer's proposed tokenizations.", required=False, choices=['most_frequent_tokens', 'ranked_tokenizations_by_word'], default=[])
    parser.add_argument('-I', '--interactive', type=str2bool, help="Run in interactive mode.. after training, the system will wait for user input to tokenize.", required=False, default=False)
    parser.add_argument('-D', '--debug', type=str2bool, help="Compute tokenizations in debug mode.", required=False, default=False)

    args = parser.parse_args()
    

    # 0) Initialize the analyzer
    stderr.write('\nInitializing analyzer with database "{}"..\n'.format(args.database))
    analyzer = Analyzer(args.database, args.separator)


    # 1) Load or train a disambiguator
    disambiguator_file = get_disambiguator_filename(args)
    try:
        stderr.write('\nTrying to load a previously trained disambiguator model:\n\t"{}"\n'.format(os.path.basename(disambiguator_file)))
        disambiguator = pickleIn(disambiguator_file)
    except FileNotFoundError:
        stderr.write('\nNone found.. Training disambiguator on "{}"..\n'.format(args.train))
        if args.clitic_factorization == 'simple':
            disambiguator = Disambiguator_simpleFactorization(analyzer, args.separator, args.ignore_class, args.clitic_factorization)
        elif args.clitic_factorization == 'complex':
            disambiguator = Disambiguator_complexFactorization(analyzer, args.separator, args.ignore_class, args.clitic_factorization)
        elif args.clitic_factorization == 'joint':
            disambiguator = Disambiguator_jointFactorization(analyzer, args.separator, args.ignore_class, args.clitic_factorization)
        disambiguator.get_possible_tokenization_statistics(args.train)
        stderr.write('\nCaching trained disambiguator..\n')
        pickleOut(disambiguator, disambiguator_file)


    # 2) Apply the trained disambiguator and write out tokenization
    if args.interactive:
        disambiguator.interact(debug=args.debug)
    stderr.write('\nApplying maximum likelihood greedy tokenization..\n\tReading input from "{}" and writing output to: "{}"..\n'.format(args.test, args.output))
    disambiguator.apply_tokenization(args.test, args.output, baseline=args.baseline, debug=args.debug)

    # 3) Print out any requested information regarding the model
    if 'most_frequent_tokens' in args.print_options:
        disambiguator.print_most_frequent_tokens()
    if 'ranked_tokenizations_by_word' in args.print_options:
        disambiguator.print_ranked_tokenizations_by_word()
    if args.print_options == None:
        stderr.write('\n')
        os.system('clear')
    stderr.write('Done! _\n     / \\ \n    /   \\    \n __/_____\\__\n   /     \\\n  | o   o |\n| |   v   | |\n| | \\___/ | |\n|  \\_____/  |\n \\____|____/\n      |\n      |\n      |\n      |\n      |\n     / \\\n    /   \\\
    \n   /     \\\n  |       |\n  |       |\n  |       |\nSo yeah, your tokenized output is here: "{}"\n\n'.format(args.output))
