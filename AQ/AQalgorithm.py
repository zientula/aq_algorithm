
#File containing the AQ algorithm logic

import numpy as np
import pandas as pd

class AQ:
    """
    This is the classs that we base of all our AQ operations on.
    """
    _dataPath = 'Data/'  # Path to the directory with our data files to be analyzed by the algorithm
    _E = []  # This is what was defined in "Systemy uczace sie" as zbior P
    _selectors = []  # All of the selectors for a given datatype with z presentation of (attribute, value)

    def __init__(self, star_max_size):
        self.data = None  # This is our zbior trenujacy T
        self.star_max_size = star_max_size  # This is the "m" parameter

    def fit(self, file_name):
        """
        This is the main function, that is called on a given dataset to create the rules for it - to learn it.
        It is used to create the list of rules "R" that will be later returned.
        :param file_name - the name of the file with data that we want to analyze. It should have the format of the
        csv file, but not necessarily the .csv extension. The file should be located in Data/ directory
        """
        self.data = pd.read_csv(self._dataPath + file_name)  # Loading data from file to zbior trenujacy T
        self._E = self.data.copy()  # At first, P is a copy of T
        self.compute_selectors()  # We use compute_selectors() to retrieve all possible selectors from datafile

        rule_list = []  # Initialization of the rule list
        classes = self.data.loc[:, [list(self.data)[-1]]]
        classes_count = classes.iloc[:, 0].value_counts()

        # We do this loop until all of the examples from the "P" (self._E) set have been covered by some rule
        while len(self._E) > 0:
            # We call on the function described in "Systemy uczace sie" as znajdz-kompleks-aq
            # It also returns the class of the returned rule (complex) alongside the negative class because it is easier for the program to process
            best_aq_cpx, p_class, n_class = self.find_aq_complex()
            # We decided to print every new rule generated
            print("New rule generated: " + str(best_aq_cpx) + " ----> " + p_class)
            if best_aq_cpx is not None:
                # We get all the examples that were covered by the rule in the P data set
                all_covered_examples = self.get_covered_examples(self._E, best_aq_cpx)
                print('')
                # We remove those examples from the P set as they were covered by the current rule
                self._E = self.remove_examples(self._E, all_covered_examples)
                # We append the rule to our rule list as a complex with the class that it represents
                rule_list.append((best_aq_cpx, p_class))

            else:
                pass

        return rule_list

    def compute_selectors(self):
        """
        Function made to compute all the possible selectors for our data set.
        It parses through all the attributes, and finds for them all possible values in the data set
        """
        attributes = list(self.data)
        del attributes[-1]

        for a in attributes:
            possible_values = set(self.data[a])
            for value in possible_values:
                self._selectors.append((a, value))

    def remove_examples(self, all_examples, indexes):
        """
        Function used to remove covered examples from the P set.
        :param all_examples - is the set of examples from which we will be removing examples, in this case - the P set
        :param indexes - indexes of examples to be removed
        We return the remaining examples.
        """
        remaining_examples = all_examples.drop(indexes)
        return remaining_examples

    def get_covered_examples(self, all_examples, best_cpx):
        """
        Function used to see which examples are covered by a complex.
        :param all_examples - all examples that are to be considered as covered or not
        :param best_cpx - the complex which will be tested as to which examples from the set it covers
        It returns the indexes of examples from all_examples that are covered by the best_cpx
        """
        values = dict()
        [values[t[0]].append(t[1]) if t[0] in list(values.keys())
         else values.update({t[0]: [t[1]]}) for t in best_cpx]
        for attribute in list(self.data):
            if attribute not in values:
                values[attribute] = set(self.data[attribute])

        covered_examples = all_examples[all_examples.isin(values).all(axis=1)]
        return covered_examples.index

    def get_unindexed(self, all_examples, best_cpx):
        """
        This function is the same as get_covered_examples, but it returns the examples as a dataframe, not their indexes.
        """
        values = dict()
        [values[t[0]].append(t[1]) if t[0] in list(values.keys())
         else values.update({t[0]: [t[1]]}) for t in best_cpx]
        for attribute in list(self.data):
            if attribute not in values:
                values[attribute] = set(self.data[attribute])

        covered_examples = all_examples[all_examples.isin(values).all(axis=1)]
        return covered_examples

    def find_aq_complex(self):
        """
        It corresponds to znajdz-kompleks-aq from "Systemy uczace sie" and contains a loop that generates a star that
        does not cover any examples from class different than that of its positive seed. The star contains each time the number
        of complexes defined by the m parameter also known as max_star_size. After the perfect star is generated, the loop ends
        and the best graded complex from the star is returned as our new rule alongside with class of positive and negative seeds.
        """

        aq_complex = None  # Initialization of the complex that will be returned
        star = []  # Initialization of the star - the star, although it might seem empty, covers all examples if
        # called with get_covered_examples(self.data, star)
        seed = self.find_positive_seed()  # Get the positive seed

        classes, c_types = self.get_classes(self.data.index)  # we use this to get c_types, that gives a list of class names, e.g. [T, F],
                                                              # and also to get the classes of examples to check whether star covers negative
                                                              # examples and to get the negative seed
        seed_class = seed[-1]  # getting seed class from seed
        n_class = [x for x in list(c_types) if x is not seed_class][0]  # getting the negative class (the one that is not the class of our current seed)
        p_class = [x for x in list(c_types) if x is seed_class][0]  # getting the positive class

        neg_class = classes.loc[classes.iloc[:, 0] == n_class]  # indexes of examples of negative class
        pos_class = classes.loc[classes.iloc[:, 0] == p_class]

        # We start the loop of generetaing the star that revolves around our seed and doesn't cover any examples from the negative class
        while not self.is_neg_not_covered(neg_class, star):
            neg_seed = self.get_negative_seed(neg_class, star)  # getting the negative seed
            part_star = self.part_star(seed, neg_seed)  # generating the partial star based on our seed and negative seed

            if len(part_star) == 0:  # if the partial star is empty, we end the loop
                # normally, it should return None, but we decided to create a rule out of a seed -> in some
                # datasets it might happen that the two identical examples have different classes, so we have to do
                # something since the seed is not chosen randomly, but the first one available
                atr = list(self.data)
                del atr[-1]
                special_case = list(zip(atr, seed))
                return special_case, p_class, n_class

            star = self.intersection(star, part_star)  # we intersect the current star with created partial star
            star = self.generalize(star)  # we perform generalization operations on the star
            star = self.decr_w_max_star(star, n_class, p_class)  # we grade every complex of the star and choose only the "m" best ones
        best_cpx = self.choose_best_cpx(star, n_class, p_class)  # after the loop: we grade complexes of the star again and coose the best one for our rule
        return best_cpx, p_class, n_class # we return the rule along with the positive and negative classes names

    def choose_best_cpx(self, star, n_class, p_class):
        """
        This function grades the complexes of the star based on given criteria and
        returns the best one.
        :param star - the star generated after the loop of find_aq_complex ends
        :param n_class - class name different than the seed's class
        :param p_class - class name of seed's class
        """
        t = self.data  # T set
        p = self._E  # P set
        eq = 0  # score for how many examples out of T that have the same class as seed are covered
        neq = 0  # score for how many examples out of T that have different class than seed are not covered
        dot_eq = 0  # score for how many examples out of P that have the same class as seed are covered
        rating = 0
        rating_table = []  # array of all complexes scores
        for cmpx in star:
            eq = self.count_covered_positive(t, cmpx, p_class)
            neq = self.count_covered_negative(t, cmpx, n_class)
            dot_eq = self.count_covered_positive(p, cmpx, p_class)
            rating = eq + neq + dot_eq
            rating_table.append(rating)

        arr = np.array(rating_table)
        arr = arr.argsort()[-1:][::-1]  # the index of complex with the best score
        return star[arr[0]]

    def decr_w_max_star(self, star, n_class, p_class):
        """
        Works the same as choose_best_cpx, but instead of one complex we choose as many as the m parameter.
        We also return a modified star, not a single complex.
        """
        t = self.data
        p = self._E
        eq = 0
        neq = 0
        dot_eq = 0
        rating = 0
        rating_table = []
        for cmpx in star:
            eq = self.count_covered_positive(t, cmpx, p_class)
            neq = self.count_covered_negative(t, cmpx, n_class)
            dot_eq = self.count_covered_positive(p, cmpx, p_class)
            rating = eq + neq + dot_eq
            rating_table.append(rating)

        if len(rating_table) < self.star_max_size:  # if generated star has less complexes than m, then we leave the star as it is
            return star

        arr = np.array(rating_table)
        arr = arr.argsort()[-self.star_max_size:][::-1] # indexes of m best graded complexes
        decreased_star = []
        for x in arr:  # we append those best complexes to a new, decreased star
            decreased_star.append(star[x])
        return decreased_star

    def count_covered_positive(self, t, cmp, p_class):
        """
        Function for counting how many of covered examples from T that have
        the same class as seed, are covered by the complex.
        :param t - T set (can be also used with P set)
        :param cmp - the complex that we are testing
        :param p_class - the class of the seed
        """
        cov = self.get_unindexed(t, cmp).iloc[:, -1]  # We get all examples from T that are covered by the complex
        p_class_counter = 0
        for x in cov:  # we count how many of the covered examples have the same class as seed
            if x == p_class:
                p_class_counter += 1
        return p_class_counter

    def count_covered_negative(self, t, cmp, n_class):
        """
        Function for counting how many of the examples from T that have different
        class than seed, have not been covered by the complex.
        :param t - T set
        :param cmp - tested complex
        :param n_class - class different than that of seed
        """
        cov = self.get_unindexed(t, cmp).iloc[:, -1]  # we get all examples from T covered by the complex
        all_ = t.iloc[:, -1]  # we get all examples from T
        n_cov_counter = 0
        n_all_counter = 0
        for x in cov:  # we count how many of the covered examples have the class different than seed
            if x == n_class:
                n_cov_counter += 1
        for x in all_:  # we count how many of all examples have the class different than seed
            if x == n_class:
                n_all_counter += 1
        # we return the number of all negative examples minus those negative covered by the complex
        return n_all_counter - n_cov_counter

    def generalize(self, star):
        """
        Function to perform the generalization operations on the given star.
        :param star - star to be generalized
        Returns the new generalized star.
        """
        attributes = list(self.data)
        del attributes[-1]
        # this loop is to remove all those complexes that might have empty selectors because of the previous intersection
        for c in star:
            for a in attributes:
                count = 0
                for s in c:
                    if s[0] == a:
                        count = count + 1
                if count == 0:
                    star.remove(c)

        # in this part of the code, the complexes that are fully covered by more general complexes are removed
        cmplx_to_remove = []
        for x in range(0, len(star)):
            if set(star[x]) < set(star[(x + 1) % len(star)]):
                if star[x] in cmplx_to_remove:
                    pass
                else:
                    cmplx_to_remove.append(star[x])
        for x in range(0, len(star)):
            if set(star[x]) > set(star[(x + 1) % len(star)]):
                if star[(x + 1) % len(star)] in cmplx_to_remove:
                    pass
                else:
                    cmplx_to_remove.append(star[(x + 1) % len(star)])
        for x in cmplx_to_remove:  # removal of the less general complexes
            star.remove(x)
        # dumplicate complexes (complexes that are identical) are also removed

        final_star = [list(x) for x in set(tuple(x) for x in star)]

        return final_star

    def intersection(self, star, part_star):
        """
        This function performs the operation of intersection between the current star and partial star.
        :param star - the currents star
        :param part_star - the partial star
        Returns the new intersected star.
        """
        temp_star = []  # for new intersected star
        if len(star) == 0:  # in the first use of the loop from find_aq_complex, the star has length 0, but theoretically has
                            # all the complexes, so in this case the intersection between the star and part_star will be the part_star
            temp_star = part_star
        else:
            # else we intersect each complex from star with each complex from partial star,
            # and append this new complexes to the star that will be returned
            for sc in star:
                for psc in part_star:
                    new_comp = self.intersect_complexes(sc, psc)
                    temp_star.append(new_comp)
        return temp_star

    def intersect_complexes(self, sc, psc):
        """
        Function to intersect complexes.
        :param sc - complex from star
        :param psc - complex from partial star
        """
        temp_comp = []
        # we check whether the complexes have the same attributes values - if they have the same value,
        # we append it to the new complex that we will return
        for ats in sc:
            for atps in psc:
                if ats == atps:
                    temp_comp.append(ats)
        return temp_comp

    def part_star(self, seed, neg_seed):
        """
        Function that generates the partial star based on seed and negative seed.
        :param seed - seed
        :param neg_seed - negative seed
        Returns the generated partial star.
        """
        partial_star = []  #Initialization of empty partial star
        attributes = list(self.data)
        del attributes[-1]
        xs = list(zip(attributes, seed))  # seed in the form of list with both attributes names and values
        xn = list(zip(attributes, neg_seed))  # the same for negative seed

        v = None  # initialization of values set

        for x in range(0, len(xs)): # We parse through attributes (not their values), hence it is possible to parse through length of seed
            v = self._selectors.copy() #  V contains all of the selectors - maksymalnie ogolny kompleks
            v.remove(xn[x])  # we remove from it the value of an attribute that negative seed has
            if xn[x] != xs[x]: # we append the selector to partial star only if the attribute value of seed was                                         # than that of
                partial_star.append(v)  # different than attribute value of negative seed (so ai(xs) belongs to V)

        return partial_star

    def get_negative_seed(self, neg_class, star):
        """
        Function to get the negative seed.
        :param neg_class - class that the negative seed should be of
        :param star - star that covers te future negative seed
        Returns the negative seed as a list.
        """
        t = self.data
        index_list = neg_class.index.tolist()
        nc = None
        if len(star) == 0:  # in the first use of the loop the star is full but looks empty, so we needed to do this distinction
            # the negative seed will be the first example of negative class covered by the star
            nc = self.get_covered_examples(t.iloc[index_list], star)
            return list(t.loc[nc[0]])
        else:
            # after the first loop we parse through the star with complexes - if the first complex does not cover any negative
            # examples, we look in the next complex and take the first negative example covered available
            for c in star:
                nc = self.get_covered_examples(t.iloc[index_list], c)
                if not nc.empty:
                    return list(t.loc[nc[0]])
        return list(t.loc[nc[0]])

    def is_neg_not_covered(self, neg_class, star):
        """
        Function that checks whether there are any examples of class different than that of seed's that the star covers.
        :param neg_class - class different than that of seed
        :param star - star
        Returns false if star covers any negative examples from T;
        returns True if there is no examples of negative class covered by the star.
        """
        t = self.data
        index_list = neg_class.index.tolist()

        if len(star) == 0:  # like before, the first use of the loop has different form of star
            covered_examples = self.get_covered_examples(t.iloc[index_list], star)
            return covered_examples.empty
        else:
            # we parse through all complexes in the star to check whether there is one that still covers negative examples
            for c in star:
                covered_examples = self.get_covered_examples(t.iloc[index_list], c)
                if not covered_examples.empty:
                    return False
            return True

    def get_classes(self, examples):
        """
        Function to get info about classes of examples from a given set of examples.
        :param examples - examples in the form of indexed dataframe
        Returns:
            classes - class of each example
            c_types - names of the two classes that occur in the example set
        """
        classes = self.data.loc[examples, [list(self.data)[-1]]]
        c_types = classes.iloc[:, 0].unique()
        return classes, c_types

    def find_positive_seed(self):
        """
        Function that gets and returns the positive seed.
        The seed is chosen as first example from the P set.
        """
        p = self._E  # The P set
        p_seed = p.iloc[0, :].values.tolist()
        return p_seed

    def test_and_stats(self, file_name, ruleset):
        """
        This function checks the mistakes matrix, True Positive Rate, True Negative Rate, and precision
        against the rules previously created and a new set of examples to test the rules on.
        :param file_name - name of the file that contains new examples to test the rules on
        :param ruleset - previously genereated rules that will be tested against new data
        Doesn't return anything - prints statistics on the console.
        """
        self.data = pd.read_csv(self._dataPath + file_name)  # getting new data/examples from the test file
        classes, c_types = self.get_classes(self.data.index) # getting classes of examples and class names
        self._E = self.data.copy()
        sp = file_name.split(".")
        name = sp[0]
        p_type = c_types[0]  # we decide that one class name will be a positive
        n_type = c_types[1]  # and the other one will be a negative
        tp_counter = 0  # true positive counter
        tn_counter = 0  # true negative counter
        fp_counter = 0  # false positive counter
        fn_counter = 0  # false negative counter
        # we check each rule against the new examples and sum up the results of the occurences of TP, TN, FP, and FN
        for rule in ruleset:
            covered_examples = self.get_unindexed(self._E, rule[0])
            for ce in covered_examples.iloc[:, -1]:
                if ce == rule[1]:
                    if ce == p_type:
                        tp_counter = tp_counter + 1
                    elif ce == n_type:
                        tn_counter = tn_counter + 1
                elif ce != rule[1]:
                    if ce == p_type:
                        fp_counter = fp_counter + 1
                    elif ce == n_type:
                        fn_counter = fn_counter + 1
            self._E = self.remove_examples(self._E, covered_examples.index)
        # Printing the statistics
        print(f'================= Statistics for {name} =================')
        print(f' Positive class is: {p_type}   Negative class is: {n_type}')
        print(f'                   Mistakes matrix                   ')
        print(f'          TN: {tn_counter}                 FP: {fp_counter}')
        print(f'          FN: {fn_counter}                 TP: {tp_counter}')
        print('')
        tpr = (tp_counter) / (tp_counter + fn_counter)  # Formula for TPR
        print(f'          TPR: {round(tpr, 2)}')  # We round up each result to to points after coma as was asked
        fpr = (fp_counter) / (tn_counter + fp_counter)  # Formula for FPR
        print(f'          FPR: {round(fpr, 2)}')
        precision = (tp_counter) / (fp_counter + tp_counter)  # Formula for precision
        print(f'    Precision: {round(precision, 2)}')
