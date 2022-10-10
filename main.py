import sys
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from statistics import mean
import matplotlib.pyplot as plt


def check_if_dna(dna_seq):
    """This function checks if the given dna_seq argument is a valid dna sequence by checking that every letter is one
    of the four possible DNA nucleotides."""
    assert isinstance(dna_seq, str), f"given dna sequence isn't a string"
    char_list = ["A", "a", "T", "t", "C", "c", "G", "g"]  # a list of the letters that can appear in a dna sequence
    for n in dna_seq:
        assert n in char_list, f"The char '{n}' in sequence {dna_seq} isn't a nucleotide found in the DNA sequences."
    # the loop goes over the dna sequence, char by char, and checks if it's in the char_list above. if not, it throws
    # assertion error and specifies the char that is problematic.


def check_if_rna(rna_seq):
    """This function checks if the given dna_seq argument is a valid dna sequence by checking that every letter is one
        of the four possible RNA nucleotides."""
    assert isinstance(rna_seq, str), f"given dna sequence isn't a string"
    char_list = ["A", "a", "U", "u", "C", "c", "G", "g"]  # a list of the letters that can appear in a dna sequence
    for n in rna_seq:
        assert n in char_list, f"The char '{n}' in sequence {rna_seq} isn't a nucleotide found in the RNA sequences."
    # the loop goes over the rna sequence, char by char, and checks if it's in the char_list above. if not, it throws
    # assertion error and specifies the char that is problematic.


class Polymerase:
    """This class represents a Polymerase that is capable of transcribing a give dna sequence while generating mutations
    given it's error rate."""

    def __init__(self, type, error_rate=0.0):
        assert type == "RNA" or type == "DNA", "Polymerase type is not correct"
        self.type = type
        self.match = {"T": "A", "A": "U", "C": "G", "G": "C"} if self.type == "RNA" \
            else {"T": "A", "A": "T", "C": "G", "G": "C"}
        assert 0 <= error_rate <= 1, "Polymerase's error rate is not between 0 and 1"
        self.error_rate = error_rate

    def transcribe(self, dna_seq):
        """This function transcribes a given dna sequence while generating mutations. It does so by randomly choosing k
        indexes that are within the given dna sequence's length, in which a mutation will occur."""
        check_if_dna(dna_seq)
        transcript = ""  # initialize an empty string
        k = int(np.ceil(len(dna_seq) * self.error_rate))  # amount of errors
        counter = 0
        indexes = []
        while counter < k:  # a loop that will randomly pick k indexes in which a mutation will occur
            i = np.random.choice(len(dna_seq))
            if i not in indexes:  # checks if the index hasn't been picked yet
                indexes.append(i)
                counter += 1
        indexes.sort()
        for i, c in enumerate(dna_seq):  # goes over the dna sequence, nucleotide by nucleotide, and adds to the
            # transcript the corresponding nucleotide using a dictionary.
            if i not in indexes:  # checks if i is an index that was randomly picked to have a mutation
                transcript += self.match[c]
            else:
                while 1:
                    # randomly picks a nucleotide from the match dictionary
                    mutation = np.random.choice(list(self.match.values()))
                    # checks if the mutation randomly picked is equal to the corresponding nucleotide, meaning it's an
                    # the correct nucleotide and won't result in a mutation, therefore we need to pick a different one.
                    if mutation != self.match[c]:
                        transcript += mutation
                        break
        return transcript[::-1]  # returns the rna sequence in reverse order so it will be in the 5'-3' orientation

    def reverse_sequence(self, dna_seq):
        """This function returns the reverse orientation of a given dna sequence. At the moment it is used to return a
        mutated genome with the right orientation in the mutant cell class."""
        check_if_dna(dna_seq)
        transcript = ""  # initialize an empty string
        for c in dna_seq:
            transcript += self.match[c]
        return transcript[::-1]


class Ribosome:
    """This class represents a Ribosome that can translate a given dna sequence to an rna sequence using it's genetic
    code, and eventually translate that rna sequence into an AA chain."""

    def __init__(self, genetic_code, start_codons):
        self.genetic_code = genetic_code
        self.start_codons = "AUG"

    def dna_to_prot(self, dna_sequence):
        check_if_dna(dna_sequence)  # make sure this is dna sequence
        dna = Seq(dna_sequence)  # convert to Seq from biopython to use its functions
        rna_seq = dna.transcribe().reverse_complement()  # convert to rna sequence
        counter = rna_seq.count_overlap("AUG")  # count number of times start codon appears in sequence

        start_codon_index = rna_seq.find("AUG")  # index of the first location of start codon
        if start_codon_index == -1:  # that means there isn't any start codons
            return "Non-coding RNA"  # since this is non coding rna
        start_codons_indexes = [start_codon_index]  # list of all indexes of start codons
        for i in range(counter):  # find all the indexes
            k = rna_seq.find("AUG", start=start_codon_index + 1)  # after finding first start codon, keep searching
            # for more start codons begin from the location of the last start codon you find
            if k == -1:  # that means there is no more start codons at the sequence
                break
            start_codons_indexes.append(k)
            start_codon_index = k

        max_prot = ""
        n = len(dna)
        for j in start_codons_indexes:  # go through all start codons
            prot = dna[:n - j].transcribe().reverse_complement().translate(to_stop=True)  # translate to protein
            if len(prot) > len(max_prot):
                max_prot = prot

        if len(max_prot) == 0:
            return "Non-coding RNA"
        return max_prot


class Cell:
    """This class represents a Cell that can undergo mitosis and meiosis, generate it's own repertoire containing it's
    dna sequences, their corresponding rna sequences and the proteins they code to, and find an srr."""

    def __init__(self, name, genome, num_copies, genetic_code, start_codons, division_rate, error_rate=0.0):
        self.name = name
        for i in range(len(genome)):
            check_if_dna(genome[i])
        self.genome = genome
        assert num_copies > 0, "num_copies isn't greater than 0"
        assert isinstance(num_copies, int), "num_copies isn't an integer"
        self.num_copies = num_copies
        self.genetic_code = genetic_code
        self.start_codons = start_codons
        assert division_rate > 1, "division_rate isn't greater than 1"
        assert isinstance(division_rate, int), "division_rate isn't an integer"
        self.division_rate = division_rate
        self.dna_polymerase = Polymerase("DNA", error_rate)
        self.rna_polymerase = Polymerase("RNA", 0.0)
        self.ribosome = Ribosome(genetic_code, start_codons)

    def __repr__(self):
        return f"<{self.name}, {self.num_copies}, {self.division_rate}>"

    def copy(self):
        """this function creates a copy of self, so that in the future if we want to edit one cell it wouldn't affect
        all copies of the cell"""
        return Cell(self.name, self.genome, self.num_copies, self.genetic_code, self.start_codons,
                    self.division_rate)

    def mitosis(self):
        """returns a list with identical copies of the cell the amount of division_rate"""
        return [self.copy() for i in range(self.division_rate)]

    def __mul__(self, other):
        """returns a list with identical copies of the cell by the amount specified with the other variable"""
        assert isinstance(other, int), f"{other} is of type {type(other)} and not an int"
        return [self.copy() for i in range(other)]

    def meiosis(self):
        """This function first checks if the cell can undergo meiosis, and if it can, it creates 2 cells with half the
        number of copies of the genome, in which one of the cells has the original genome, and the other has the
        complementary genome."""
        if self.num_copies % 2 != 0:
            return None
        first = Cell(self.name, self.genome, self.num_copies / 2, self.genetic_code, self.start_codons,
                     self.division_rate)
        complementary_genome = [self.dna_polymerase.transcribe(sequence) for sequence in self.genome]
        # line above transcribes the complementary sequence to all sequences in the original genome
        second = Cell(self.name, complementary_genome, self.num_copies / 2, self.genetic_code, self.start_codons,
                      self.division_rate)
        return [first, second]


class ProkaryoticCell(Cell):
    def __init__(self, genome, error_rate=0.0):
        prokaryotic_genetic_code = {
            'AUA': 'I', 'AUC': 'I', 'AUU': 'I', 'AUG': 'M',
            'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACU': 'T',
            'AAC': 'N', 'AAU': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGC': 'S', 'AGU': 'S', 'AGA': 'R', 'AGG': 'R',
            'CUA': 'L', 'CUC': 'L', 'CUG': 'L', 'CUU': 'L',
            'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCU': 'P',
            'CAC': 'H', 'CAU': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGU': 'R',
            'GUA': 'V', 'GUC': 'V', 'GUG': 'V', 'GUU': 'V',
            'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCU': 'A',
            'GAC': 'D', 'GAU': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGU': 'G',
            'UCA': 'S', 'UCC': 'S', 'UCG': 'S', 'UCU': 'S',
            'UUC': 'F', 'UUU': 'F', 'UUA': 'L', 'UUG': 'L',
            'UAC': 'Y', 'UAU': 'Y', 'UAA': None, 'UAG': None,
            'UGC': 'C', 'UGU': 'C', 'UGA': 'U', 'UGG': 'W'}
        super().__init__("ProkaryoticCell", genome, 1, prokaryotic_genetic_code, ["AUG", "GUG", "UUG"], 4, error_rate)


class EukaryoticCell(Cell):
    def __init__(self, name, genome, division_rate, error_rate=0.0):
        standard_genetic_code = {
            'AUA': 'I', 'AUC': 'I', 'AUU': 'I', 'AUG': 'M',
            'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACU': 'T',
            'AAC': 'N', 'AAU': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGC': 'S', 'AGU': 'S', 'AGA': 'R', 'AGG': 'R',
            'CUA': 'L', 'CUC': 'L', 'CUG': 'L', 'CUU': 'L',
            'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCU': 'P',
            'CAC': 'H', 'CAU': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGU': 'R',
            'GUA': 'V', 'GUC': 'V', 'GUG': 'V', 'GUU': 'V',
            'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCU': 'A',
            'GAC': 'D', 'GAU': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGU': 'G',
            'UCA': 'S', 'UCC': 'S', 'UCG': 'S', 'UCU': 'S',
            'UUC': 'F', 'UUU': 'F', 'UUA': 'L', 'UUG': 'L',
            'UAC': 'Y', 'UAU': 'Y', 'UAA': None, 'UAG': None,
            'UGC': 'C', 'UGU': 'C', 'UGA': None, 'UGG': 'W'}
        super().__init__(name, genome, 2, standard_genetic_code, ["AUG"], division_rate, error_rate)


class NeuronCell(EukaryoticCell):
    def __init__(self, genome):
        super().__init__("NeuronCell", genome, division_rate=2)


class StemCell(EukaryoticCell):
    def __init__(self, genome):
        super().__init__("StemCell", genome, division_rate=3, error_rate=0.0)


class MutantCell(StemCell):
    """This function inherits StemCell, and has it's own mitosis and mul functions in which they generate mutations,
    and if the amount of mutations surpasses 10 - they create a cancer cell."""

    def __init__(self, genome, num_mutations=0, error_rate=0.05):
        super().__init__(genome)
        self.num_mutations = num_mutations
        self.name = "MutantCell"
        self.dna_polymerase = Polymerase("DNA", error_rate)

    def mitosis(self):
        """returns a list with one identical copy of the cell, and the rest are cells with more mutations"""
        mutations = self.num_mutations
        mutated_genome = []
        cells = []
        for dna_seq in self.genome:
            mutated = self.dna_polymerase.transcribe(dna_seq)  # transcribes a mutated genome for each
            mutated_genome.append(self.dna_polymerase.reverse_sequence(mutated))
            # line above is to get the original orientation of the genome back, and to append it to mutated_genome.
            mutations += int(np.ceil(self.dna_polymerase.error_rate * len(dna_seq)))
        for i in range(self.division_rate - 1):
            if mutations > 10:
                cells.append(CancerCell(mutated_genome, mutations))
            else:
                cells.append(MutantCell(mutated_genome, mutations))
        return cells

    def __mul__(self, other):
        """returns a list with identical copies of the cell by the amount specified with the other variable"""
        assert isinstance(other, int), f"{other} is of type {type(other)} and not an int"
        mutated_genome = []
        cells = []
        mutations = self.num_mutations
        for dna_seq in self.genome:
            mutated = self.dna_polymerase.transcribe(dna_seq)  # transcribes a mutated genome for each
            mutated_genome.append(self.dna_polymerase.reverse_sequence(mutated))
            mutations += int(np.ceil(self.dna_polymerase.error_rate * len(dna_seq)))
            # line above is to get the original orientation of the genome back, and to append it to mutated_genome.
        for i in range(self.division_rate - 1):
            if mutations > 10:
                cells.append(CancerCell(mutated_genome, mutations))
            else:
                cells.append(MutantCell(mutated_genome, mutations))
        return cells


class CancerCell(MutantCell):
    def __init__(self, genome, num_mutations):
        super().__init__(genome)
        self.name = "CancerCell"
        self.division_rate = 10
        self.num_mutations = num_mutations


def factory(name, genome):
    valid_names = ["StemCell", "NeuronCell", "ProkaryoticCell", "MutantCell"]
    assert name in valid_names, "Invalid cell name!"
    if name == "StemCell":
        return StemCell(genome)
    elif name == "NeuronCell":
        return NeuronCell(genome)
    elif name == "ProkaryoticCell":
        return ProkaryoticCell(genome)
    elif name == "MutantCell":
        return MutantCell(genome)


def count_proteins(cells):
    unique_proteins = set()  # to keep only the unique proteins
    for cell in cells:  # go through all cells
        for seq in cell.genome:  # go through all sequence the cell contains
            protein = cell.ribosome.dna_to_prot(seq)  # translate the sequence to protein
            if protein != "Non-coding RNA":  # if the sequence was encoding to protein
                unique_proteins.add(protein)  # add the protein to the list
    return unique_proteins


def single_simulation(sequence, divison_cycle, error_rate):
    """this function runs 3 simulations of the same parameters and calculates the average of the results
     in order to maintain correctness"""

    count_mutant_cells = []  # list for the number of mutant cells at each repeat, to calculate average after
    count_cancer_cells = []   # list for the number of cancer cells at each repeat, to calculate average after
    count_unique_proteins = []   # list for the number of unique proteins at each repeat, to calculate average after
    for i in range(3):  # for 3 repeats for every experiment
        orig_cell = MutantCell(sequence, 0, error_rate)  # the original cell that will divide
        cells = [orig_cell]  # list of all cells, the first cell is the first in the list
        for j in range(divison_cycle):
            new_cells = []  # to store all the cells that derived from the cells that exist now
            for cell in cells:  # for every cell in the list, divide the cell thorough mitosis
                new_cells += cell.mitosis()  # put all the cells that you got from mitosis in the new cells list
            cells += new_cells  # add the new cells to the list of all cells
        count_mut = 0  # to count the number of mutant cells
        count_can = 0  # to count the number of cancer cells
        for cell in cells:
            if cell.name == "MutantCell":
                count_mut += 1
            elif cell.name == "CancerCell":
                count_can += 1
        count_mutant_cells.append(count_mut)
        count_cancer_cells.append(count_can)
        proteins_num = int(len(count_proteins(cells)))  # the amount of unique proteins that the cells contains
        count_unique_proteins.append(proteins_num)
    average_mutant_cells = mean(count_mutant_cells)
    average_cancer_cells = mean(count_cancer_cells)
    average_proteins = mean(count_unique_proteins)

    return average_mutant_cells, average_cancer_cells, average_proteins


def full_simulation(sequences, sequences_names):
    """this function runs full simulation"""

    error_rates_array =[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    division_cycles_array = [1, 2, 3, 4, 5]
    single_sequence_frame = []  # for

    for i, seq in enumerate(sequences):  # for each sequence, create data for csv
        frames = []  # frames for csv to each cycle
        names = [sequences_names[i]] * len(error_rates_array)
        for cycle in division_cycles_array:
            average_mutant_cells = []
            average_cancer_cells = []
            average_proteins = []
            for err in error_rates_array:
                single_simulate = single_simulation(seq, cycle, err)  # return the average of 3 repeats
                average_mutant_cells.append(single_simulate[0])
                average_cancer_cells.append(single_simulate[1])
                average_proteins.append(int(single_simulate[2]))
            df = pd.DataFrame({
                    "Sequence Name": names, "Cycle": [cycle] * len(error_rates_array), "Error Rate": error_rates_array,
                    "Mutant Cells": average_mutant_cells, "Cancer Cells": average_cancer_cells,
                    "Unique Proteins": average_proteins
                })
            frames.append(df)
        single_sequence_frame.append(pd.concat(frames))  # all data to store for single sequence
    all_frames = pd.concat(single_sequence_frame)
    all_frames.to_csv(r'./exercise4_208948927_315535351.csv', index=False)


def graphs(sequence_name):
    """this function generates graphs for all the data"""

    data = pd.read_csv('./exercise4_208948927_315535351.csv')
    relevant_seq = data.loc[data['Sequence Name'] == sequence_name]  # to extract the data relevant for the sequence
    max_cycle = relevant_seq.loc[relevant_seq["Cycle"] == 5]  # first 3 graphs are for the maximum cycle - 5
    error_rates = max_cycle.loc[:, "Error Rate"]  # extract error rates column
    mutants = max_cycle.loc[:, "Mutant Cells"]  # extract mutant cells column
    cancer = max_cycle.loc[:, "Cancer Cells"]  # extract cancer cells column
    proteins = max_cycle.loc[:, "Unique Proteins"]  # extract proteins column

    plt.figure(0)  # graph of correlation between error rates and mutant cells amount
    plt.plot(error_rates, mutants, color = 'c')
    plt.xlabel("Error Rate")
    plt.ylabel("Mutant Cell Amount")
    plt.title("Mutant cells for " + sequence_name)
    plt.savefig('ex4_208948927_315535351 ' + sequence_name + 'mutants_error.png')
    plt.close(0)

    plt.figure(1)  # graph of correlation between error rates and cancer cells amount
    plt.plot(error_rates, cancer, color='b')
    plt.xlabel("Error Rate")
    plt.ylabel("Cancer Cell Amount")
    plt.title("Cancer cells for " + sequence_name)
    plt.savefig('ex4_208948927_315535351 ' + sequence_name + 'cancer_error.png')
    plt.close(1)

    plt.figure(2)  # graph of correlation between error rates and unique proteins amount
    plt.plot(error_rates, proteins, color='m')
    plt.ylim(proteins.min() - 50, proteins.max() + 50)
    plt.xlabel("Error Rate")
    plt.ylabel("Unique Proteins Amount")
    plt.title("Unique Proteins for " + sequence_name)
    plt.savefig('ex4_208948927_315535351 ' + sequence_name + 'proteins_error.png')
    plt.close(2)

    # graph for dependency of protein amount on cycles and error rates
    relev = data.loc[data["Sequence Name"] == sequence_name]
    cycle1 = relev.loc[data["Cycle"] == 1]
    prot_cycle1 = np.array(cycle1.loc[:, ["Unique Proteins"]])
    err1 = np.array(cycle1.loc[:, ["Error Rate"]])
    cycle2 = relev.loc[data["Cycle"] == 2]
    prot_cycle2 = np.array(cycle2.loc[:, ["Unique Proteins"]])
    cycle3 = relev.loc[data["Cycle"] == 3]
    prot_cycle3 = np.array(cycle3.loc[:, ["Unique Proteins"]])
    cycle4 = relev.loc[data["Cycle"] == 4]
    prot_cycle4 = np.array(cycle4.loc[:, ["Unique Proteins"]])
    cycle5 = relev.loc[data["Cycle"] == 5]
    prot_cycle5 = np.array(cycle5.loc[:, ["Unique Proteins"]])

    plt.figure(3)
    plt.plot(err1, prot_cycle1, 'b')
    plt.plot(err1, prot_cycle2, 'k')
    plt.plot(err1, prot_cycle3, 'm')
    plt.plot(err1, prot_cycle4, 'c')
    plt.plot(err1, prot_cycle5, 'r')

    plt.xticks(err1)
    # convert y-axis to Logarithmic scale
    plt.yscale("log")
    plt.title(sequence_name)
    plt.ylabel("Proteins")
    plt.xlabel("Error Rate")
    plt.legend(["Cycle 1", "Cycle 2", "Cycle 3", "Cycle 4", "Cycle 5"], loc="upper right")

    plt.savefig('ex4_208948927_315535351 ' + sequence_name + 'all_data.png')
    plt.close(3)


def main():

    assert len(sys.argv) > 1, "Didn't receive any arguments as input"
    # even if it's greater than 1, we only use first argument
    assert sys.argv[1].endswith('.fa'), "Given system argument isn't a fasta file"
    sequences = [[str(seq_record.seq.upper())] for seq_record in SeqIO.parse(sys.argv[1], "fasta")]  # this generate
    # list of all the sequences from FASTA file
    sequences_names = [seq_record.name for seq_record in SeqIO.parse(sys.argv[1], "fasta")]  # this generate
    # list of all the sequences names from FASTA file

    full_simulation(sequences, sequences_names)
    for name in sequences_names:
        graphs(name)


if __name__ == '__main__':
    main()
