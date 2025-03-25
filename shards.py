# Author: Brian S. Cole, PhD; Yuyang Luo, PhD; Ayellet Segre, PhD\
# Massacusetts Eye and Ear, Havard Medical School
# 2020

# Modified by Hamy Vy, PhD; Icahn School of Medicine at Mount Sinai
# 2022


import re
import pdb
import time
import typing

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from random import random
from shutil import copyfile
from pandas import DataFrame, Series
from argparse import ArgumentParser, Namespace
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from seaborn import heatmap, scatterplot, clustermap, lineplot
from subprocess import run, CompletedProcess, CalledProcessError

def get_args() -> Namespace:
    """
    Parse command line arguments using argparse.
    Returns the parsed argparse.Namespace object.
    """
    parser = ArgumentParser()
    parser.add_argument('--study_vcf', type=str, required=True,
                        help='VCF from study subjects to infer ancestry')
    parser.add_argument('--reference_vcf', type=str, required=True,
                        help='VCF from reference panel')
    parser.add_argument('--reference_populations', type=str, required=True,
                        help='TSV of populations from reference panel. '
                             'Needs first column as sample ID, '
                             'second column as population')
    parser.add_argument('--output_file', type=str,
                        default='inferred_ancestry.tsv',
                        help='Output TSV of inferred ancestry for study')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose operation')
    parser.add_argument('--min_maf', type=float, default=0.05,
                        help='Minimum MAF for defining common variants. '
                        'Default: 0.05')
    parser.add_argument('--max_missingness', type=float, default=0.01,
                        help='Maximum missingness of common variants. '
                        'Default: 0.01')
    parser.add_argument('--max_r2', type=float, default=0.2,
                        help='Maximum R^2 for LD pruning via Plink')
    parser.add_argument('--window_size', type=int, default=200,
                        help='Window size for LD pruning via Plink')
    parser.add_argument('--step_size', type=int, default=100,
                        help='Step size for LD pruning via Plink')
    parser.add_argument('--plink_path', type=str, default='plink',
                        help='Path to plink. Needed if plink is not '
                             'in $PATH')
    parser.add_argument('--reference_vcf_has_chrom_pos_ids',
                        action='store_true',
                        help='Reference VCF has IDs of the format '
                        'chrom:pos so no need to reformat them')
    parser.add_argument('--delete_intermediate_files', action='store_true',
                        help='Delete intermediate files at end')
    parser.add_argument('--underscore_sample_id', action='store_true',
                        help='The delimiter of samples ID is underscore or not')
    parser.add_argument('--number_of_principal_components', type=int, default=20,
                        help='The maximum number of principal components for cross validation')
    parser.add_argument('--number_of_neighbors', type=int, default=50,
                        help='The maximum number of neighbors for cross validation')
    args = parser.parse_args()
    return args


def die(epitaph: str,
        exit_code: int = 1) -> None:
    """
    Print a message and exit.
    """
    print(epitaph)
    exit(exit_code)


def validate_args(args: Namespace) -> bool:
    """
    Given the parsed arguments, validate them.
    """
    files_which_need_to_exist = [args.study_vcf,
                                 args.reference_vcf,
                                 args.reference_populations]
    return all(
        [Path(f).is_file() for f in files_which_need_to_exist]
        )


def set_var_ids(bim_file: str,
                chrom_pos_id: bool = False,
                bak_prefix: str = '.bak') -> None:
    """
    Reset the variant IDs in a bim file
    to chr:pos:ref:alt in place.
    If chrom_pos_id is set to True,
    set the IDs just to chr:pos in place.
    Keeps a backup of the original bim file.
    """
    ##Chromosome, Variant identifier, Position in morgans or centimorgans (safe to use dummy value of '0')
    ##Base-pair coordinate, Allele 1, Allele 2.
    backup_bim_file = bim_file + bak_prefix
    copyfile(bim_file, backup_bim_file)
    with open(backup_bim_file) as bim_in, open(bim_file, 'w') as bim_out:
        for line in bim_in:
            chrom, snp_id, cM, pos, ref, alt = line.rstrip().split('\t')
            new_id = ':'.join(
                ['chr' + chrom, pos] if chrom_pos_id
                else ['chr' + chrom, pos, ref, alt])
            print('\t'.join([chrom, new_id, cM, pos, ref, alt]), file=bim_out)
    Path(backup_bim_file).unlink()


def bim_to_var_ids(bim_file: str,
                   var_ids_file: str) -> None:
    """
    Extract the variant IDs (second column)
    from a Plink BIM file and save it to a single column
    text file. Useful to generate a file that Plink can use
    to subset variants.
    """
    with open(bim_file) as inny, open(var_ids_file, 'w') as outy:
        for line in inny:
            fields = line.rstrip().split()
            print(f"{fields[1]}", file=outy)

def multi_allelic_ids_excluded(bim_file: str,
                      var_ids_file: str) -> None:
    """
    Extract the variant IDs (second column)
    from a Plink BIM file and save it to a single column
    text file. Useful to generate a file that Plink can use
    to subset variants.
    """
    with open(bim_file) as inny, open(var_ids_file, 'w') as outy:
        dicts = dict()
        for line in inny:
            fields = line.rstrip().split('\t')
            var_chr_pos = fields[1]
            if var_chr_pos not in dicts:
                dicts[var_chr_pos] = 1
            else:
                dicts[var_chr_pos] += 1
        #iterate dicts
        for key in dicts:
            if dicts[key] == 1:
                print(f"{key}", file=outy)

def ambiguous_site_ids_excluded(bim_file: str,
                                var_ids_file: str) -> None:
    """
    remove A/T or G/C; Extract the variant IDs (second column)
    from a Plink BIM file and save it to a single column
    text file. Useful to generate a file that Plink can use
    to subset variants.
    """
    with open(bim_file) as inny, open(var_ids_file, 'w') as outy:
        for line in inny:
            chrom, var_chr_pos, cM, pos, ref, alt = line.rstrip().split('\t')
            if ref == 'A' and alt == 'T' or ref == 'T' and alt == 'A' or ref == 'G' and alt == 'C' or ref == 'C' and alt == 'G':
                continue
            else:
                print(f"{var_chr_pos}", file=outy)

def run_plink(plink_command: str,
              verbose: bool = False,
              check: bool = True) -> CompletedProcess:
    """
    Given a string of a plink command,
    run it in a subprocess and return the CompletedProcess
    object.
    """
    if verbose:
        print(f'Calling plink:\n{plink_command}\n')
    try:
        completed_process = run(plink_command.split(),
                                capture_output=True,
                                check=check)
        return completed_process
    except CalledProcessError as called_process_error:
        die(f'Got an error from Plink:\n'
            f'{completed_process.stderr}',
            completed_process.returncode)

def run_R(R_command: str,
              verbose: bool = False,
              check: bool = True) -> CompletedProcess:
    if verbose:
        print(f'Calling R:\n{R_command}\n')
    try:
        completed_process = run(R_command.split(),
                                capture_output=True,
                                check=check)
        return completed_process
    except CalledProcessError as called_process_error:
        die(f'Got an error from R:\n'
            f'{completed_process.stderr}',
            completed_process.returncode)

def filter_variants(vcf_file: str,
                    trimmed_study_file: str,
                    min_maf: float = 0.05,
                    max_missingness: float = 0.01,
                    path_to_plink: str = 'plink',
                    verbose: bool = False,
                    underscore_sample_id: bool = False) -> CompletedProcess:
    """
    Given a VCF file, subset common, biallelic SNVs with
    minimal missingness to a new Plink fileset.
    """
    plink_command = (f'{path_to_plink} --vcf {vcf_file} --maf {min_maf} ' + ("--double-id " if underscore_sample_id else "") +
                     f'--geno {max_missingness} --snps-only --biallelic-only '
                     f'--make-bed --out {trimmed_study_file}')
    return run_plink(plink_command, verbose=verbose)

def ld_prune_variants(plink_file: str,
                      pruned_file: str,
                      r2: float = 0.2,
                      window_size: int = 200,
                      step_size: int = 100,
                      path_to_plink: str = 'plink',
                      verbose: bool = False,
                      underscore_sample_id: bool = False) -> CompletedProcess:
    """
    Call Plink to LD-prune variants in a study.
    Doesn't actually subset variants, just generates
    a prune.in file for later use.
    """
    plink_call = (f'{path_to_plink} --bfile {plink_file} ' + ("--double-id " if underscore_sample_id else "") +
                  f'--indep-pairwise {window_size} {step_size} {r2} '
                  f'--out {pruned_file}')
    return run_plink(plink_call, verbose=verbose)

def vcf_to_plink(vcf_file: str,
                 plink_file: str,
                 path_to_plink: str = 'plink',
                 verbose: bool = False,
                 underscore_sample_id: bool = False) -> CompletedProcess:
    """
    Call Plink to convert a VCF file to a Plink file.
    """
    plink_call = (f'{path_to_plink} --vcf {vcf_file} ' + ("--double-id " if underscore_sample_id else "") +
                  f'--make-bed --out {plink_file}')
    return run_plink(plink_call, verbose=verbose)

def subset_variants(file_to_subset: str,
                    subsetted_file: str,
                    variants_to_keep: str,
                    path_to_plink: str = 'plink',
                    verbose: bool = False,
                    underscore_sample_id: bool = False) -> CompletedProcess:
    """
    Call Plink to subset variants in a Plink file.
    """
    plink_call = (f'{path_to_plink} --bfile {file_to_subset} '
                  f'--extract {variants_to_keep} --snps-only ' + ("--double-id " if underscore_sample_id else "") +
                  f'--biallelic-only --make-bed --out {subsetted_file}')
    return run_plink(plink_call, verbose=verbose)

def merge_plink(first_file: str,
                second_file: str,
                merged_file: str,
                path_to_plink: str = 'plink',
                verbose: bool = False,
                underscore_sample_id: bool = False) -> CompletedProcess:
    """
    Call Plink to merge two Plink files.
    Don't use the 'check' option because this will fail
    with variants that need flipping or exclusion.
    """
    plink_call = (f'{path_to_plink} --bfile {first_file} ' + ("--double-id " if underscore_sample_id else "") +
                  f'--bmerge {second_file} --make-bed --out '
                  f'{merged_file}')
    return run_plink(plink_call, verbose=verbose,
                     check=False)

def flip_variants(plink_file: str,
                  to_flip: str,
                  path_to_plink: str = 'plink',
                  verbose: bool = False,
                  underscore_sample_id: bool = False) -> CompletedProcess:
    """
    Call Plink to flip variants specified in a text file.
    Variants are flipped in place (mutates the input file).
    """
    plink_call = (f'{path_to_plink} --bfile {plink_file} ' + ("--double-id " if underscore_sample_id else "") + 
                  f'--flip {to_flip} --make-bed --out '
                  f'{plink_file}')
    return run_plink(plink_call, verbose=verbose)

def exclude_variants(plink_file: str,
                     to_exclude: str,
                     path_to_plink: str = 'plink',
                     verbose: bool = False,
                     underscore_sample_id: bool = False) -> CompletedProcess:
    """
    Call plink to exclude variants specified in a text file.
    Variants are excluded in place (mutates the input file).
    """
    plink_call = (f'{path_to_plink} --bfile {plink_file} ' + ("--double-id " if underscore_sample_id else "") +
                  f'--exclude {to_exclude} --make-bed --out '
                  f'{plink_file}')
    return run_plink(plink_call, verbose=verbose)

def five_pass_merge(first_file: str,
                    second_file: str,
                    merged_file: str,
                    path_to_plink: str = 'plink',
                    verbose: bool = False) -> CompletedProcess:
    """
    Merge two Plink files in (up to) five passes.
    1) Try a merge. If it worked, return the merged file.
    2) Flip variants in the second_file that can't be merged.
    3) Repeat step 1.
    4) Exclude variants in the second_file that can't be merged.
    5) Repeat step 1.
    """
    merge_completed_process = merge_plink(first_file=first_file,
                                          second_file=second_file,
                                          merged_file=merged_file,
                                          path_to_plink=path_to_plink,
                                          verbose=verbose)
    if merge_completed_process.returncode == 0:
        return merge_completed_process
    else:
        if verbose:
            print('Flipping variants in merged-merge.missnp')
        flip_completed_process = flip_variants(plink_file=second_file,
                                               to_flip='merged-merge.missnp',
                                               path_to_plink=path_to_plink,
                                               verbose=verbose)
        merge_completed_process = merge_plink(first_file=first_file,
                                              second_file=second_file,
                                              merged_file=merged_file,
                                              path_to_plink=path_to_plink,
                                              verbose=verbose)
        if merge_completed_process.returncode == 0:
            return merge_completed_process
        else:
            print('Excluding variants in merged-merge.missnp')
            exclude_proc = exclude_variants(plink_file=second_file,
                                            to_exclude='merged-merge.missnp',
                                            path_to_plink=path_to_plink,
                                            verbose=verbose)
            exclude2_proc = exclude_variants(plink_file=first_file,
                                             to_exclude='merged-merge.missnp',
                                             path_to_plink=path_to_plink,
                                             verbose=verbose)
            return merge_plink(first_file=first_file,
                               second_file=second_file,
                               merged_file=merged_file,
                               path_to_plink=path_to_plink,
                               verbose=verbose)

def run_pca(plink_file: str,
            path_to_plink: str = 'plink',
            verbose: bool = False) -> CompletedProcess:
    """
    Call Plink to compute PCA.
    """
    plink_call = (f'{path_to_plink} --bfile {plink_file} '
                  f'--pca header tabs var-wts')
    return run_plink(plink_call, verbose=verbose)

def run_pcair(plink_file: str,
            verbose: bool = False) -> CompletedProcess:

    R_call = (f'Rscript PCAIR.R --genofile {plink_file} ')
    return run_R(R_call, verbose=verbose)

def read_pcs_and_populations(pcs_file: str,
                             populations_file: str) -> DataFrame:
    """
    Read in a plink.eigenvec file from PCA.
    Simultaneously, read in the populatins file from
    the reference panel.
    Return a DataFrame of PCs for the merged study and reference
    panel samples with population labels on the reference samples.
    """
    pcs_df = pd.read_csv(pcs_file, sep="\t").drop('FID', axis=1)
    populations_df = pd.read_csv(populations_file, sep="\t")
    merged = pcs_df.merge(populations_df, on='IID', how='outer')
    return merged

def read_pcair_and_populations(pcs_file: str,
                             populations_file: str) -> DataFrame:
    pcs_df = pd.read_csv(pcs_file, sep="\t")
    populations_df = pd.read_csv(populations_file, sep="\t")
    merged = pcs_df.merge(populations_df, on='IID', how='outer')
    return merged

def split_pcs(merged_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Given a DataFrame containing both reference and study samples,
    return two DataFrames: one of the study samples and one of the
    reference samples.
    """
    reference_df = merged_df[merged_df['Ancestry'].notna()]
    samples_df = merged_df[merged_df['Ancestry'].isna()]
    return samples_df, reference_df

def optimize_classifier(df: DataFrame,
                        max_n_pcs: int = 20,
                        max_n_neighbors: int = 50,
                        verbose: bool = False) -> Tuple[int, int, list, list, list]:
    """
    Sweep over the number of PCs and the number of neighbors
    and optimize a KNeighborsClassifier.
    Return the optimal number of PCs and the optimal number of neighbors.
    """
    trimmed_df = df.drop(['IID', 'Population'], axis=1)
    optimal_n_pcs, optimal_n_neighbors = 0, 0
    optimal_accuracy = 0
    y = trimmed_df['Ancestry']
    list_for_pcs=[]
    list_for_neighbors=[]
    list_for_accuracy=[]
    for n_pcs in range(2, max_n_pcs+1):
        X = trimmed_df.drop('Ancestry', axis=1).iloc[:, 0:n_pcs]
        for n_neighbors in range(5, max_n_neighbors+1, 5):
            knc = KNeighborsClassifier(n_neighbors=n_neighbors)
            knc_scores = cross_val_score(knc, X, y, cv=10,
                                         scoring='balanced_accuracy')
            this_mean_accuracy = knc_scores.mean()
            list_for_pcs.append(n_pcs)
            list_for_neighbors.append(n_neighbors)
            list_for_accuracy.append(this_mean_accuracy)
            if this_mean_accuracy > optimal_accuracy:
                optimal_n_pcs = n_pcs
                optimal_n_neighbors = n_neighbors
                optimal_accuracy = this_mean_accuracy
        if verbose:
            print(f"Done with {n_pcs} principal components.")
    print(f"Optimal number of PCs: {optimal_n_pcs}")
    print(f"Optimal number of neighbors: {optimal_n_neighbors}")
    print(f"Optimal balanced accuracy: {optimal_accuracy:.3f}")
    return optimal_n_pcs, optimal_n_neighbors, list_for_pcs, list_for_neighbors, list_for_accuracy


def main() -> None:
    args = get_args()

    # Validate arguments.
    if not validate_args(args):
        die('Invalid arguments: one or more required files '
            'do not exist.', 255)

    files_to_delete = []  # Append files to clean up at end.

    # Subset common, unlinked variants from study:
    ##This step will also conovert vcf format to bim format
    trimmed_study_file = 'trimmed_study'
    filter_variants_completed_process = filter_variants(
        args.study_vcf,
        trimmed_study_file,
        min_maf=args.min_maf,
        max_missingness=args.max_missingness,
        path_to_plink=args.plink_path,
        verbose=args.verbose,
        underscore_sample_id=args.underscore_sample_id)
    if args.delete_intermediate_files:
        # Delete the filtered variants Plink fileset later.
        files_to_delete.extend(
            ['trimmed_study.bim', 'trimmed_study.bed',
             'trimmed_study.fam', 'trimmed_study.log',
             'trimmed_study.nosex'])

    # Recode the variant IDs to "{chrom}:{pos}" for merging.
    set_var_ids('trimmed_study.bim', chrom_pos_id=True)
    # LD prune the study's variants:
    ld_pruned_study_file = 'ld_pruned_study'
    ld_prune_variants_completed_process = ld_prune_variants(
        plink_file=trimmed_study_file,
        pruned_file=ld_pruned_study_file,
        r2=args.max_r2,
        window_size=args.window_size,
        path_to_plink=args.plink_path,
        step_size=args.step_size,
        verbose=args.verbose,
        underscore_sample_id=args.underscore_sample_id)
    if args.delete_intermediate_files:
        files_to_delete.extend(
            ['ld_pruned_study.prune.in',
             'ld_pruned_study.prune.out',
             'ld_pruned_study.log',
             'ld_pruned_study.nosex'])

    # Subset the LD-pruned varaints from the reference VCF.
    # This requires the reference VCF to have similarly formatted IDs
    # and be in the same genome build.
    if not args.reference_vcf_has_chrom_pos_ids:
        reference_plink_file = 'reference'
        vcf_to_plink_completed_process = vcf_to_plink(
            vcf_file=args.reference_vcf,
            plink_file=reference_plink_file,
            path_to_plink=args.plink_path,
            verbose=args.verbose,
            underscore_sample_id=args.underscore_sample_id)
        set_var_ids('reference.bim', chrom_pos_id=True)

        if args.delete_intermediate_files:
            files_to_delete.extend(
                ['reference.bed', 'reference.fam',
                 'reference.bim', 'reference.log',
                 'reference.nosex'])

    subsetted_reference_file = 'subsetted_reference'
    subset_reference_completed_process = subset_variants(
        file_to_subset=reference_plink_file,
        subsetted_file=subsetted_reference_file,
        variants_to_keep='ld_pruned_study.prune.in',
        path_to_plink=args.plink_path,
        verbose=args.verbose,
        underscore_sample_id=args.underscore_sample_id)
    if args.delete_intermediate_files:
        files_to_delete.extend(
            ['subsetted_reference.bed',
             'subsetted_reference.fam',
             'subsetted_reference.bim',
             'subsetted_reference.nosex',
             'subsetted_reference.log'])

    ##add two new added functinos for subsetted_reference
    #remove multiallelic
    subsetted_reference_remove_multiallelic_file = 'subsetted_reference_remove_multiallelic'
    multi_allelic_ids_excluded(bim_file='subsetted_reference.bim',
                               var_ids_file='multiallelic.reference.in')
    if args.delete_intermediate_files:
        files_to_delete.append('multiallelic.reference.in')
    remove_multiallelic_refernce_completed_process = subset_variants(
        file_to_subset=subsetted_reference_file,
        subsetted_file=subsetted_reference_remove_multiallelic_file,
        variants_to_keep='multiallelic.reference.in',
        path_to_plink=args.plink_path,
        verbose=args.verbose)
    if args.delete_intermediate_files:
        files_to_delete.extend(
            ['subsetted_reference_remove_multiallelic.bed',
             'subsetted_reference_remove_multiallelic.fam',
             'subsetted_reference_remove_multiallelic.bim',
             'subsetted_reference_remove_multiallelic.nosex',
             'subsetted_reference_remove_multiallelic.log'])    
    
    #remove ambiguous
    subsetted_reference_remove_ambiguous_file = 'subsetted_reference_remove_ambiguous'
    ambiguous_site_ids_excluded(bim_file='subsetted_reference_remove_multiallelic.bim',
                               var_ids_file='ambiguous.reference.in')
    if args.delete_intermediate_files:
        files_to_delete.append('ambiguous.reference.in')
    remove_ambiguous_refernce_completed_process = subset_variants(
        file_to_subset=subsetted_reference_remove_multiallelic_file,
        subsetted_file=subsetted_reference_remove_ambiguous_file,
        variants_to_keep='ambiguous.reference.in',
        path_to_plink=args.plink_path,
        verbose=args.verbose)
    if args.delete_intermediate_files:
        files_to_delete.extend(
            ['subsetted_reference_remove_ambiguous.bed',
             'subsetted_reference_remove_ambiguous.fam',
             'subsetted_reference_remove_ambiguous.bim',
             'subsetted_reference_remove_ambiguous.nosex',
             'subsetted_reference_remove_ambiguous.log'])

    # Not all of the variants in the subsetted reference file
    # will also exist in the LD-pruned study, so now
    # reciprocally subset the LD-pruned study:
    subsetted_study_file = 'subsetted_study'
    bim_to_var_ids(bim_file='subsetted_reference_remove_ambiguous.bim',
                   var_ids_file='intersection.prune.in')
    if args.delete_intermediate_files:
        files_to_delete.append('intersection.prune.in')
    subset_study_completed_process = subset_variants(
        file_to_subset=trimmed_study_file,
        subsetted_file=subsetted_study_file,
        variants_to_keep='intersection.prune.in',
        path_to_plink=args.plink_path,
        verbose=args.verbose)
    if args.delete_intermediate_files:
        files_to_delete.extend(
            ['subsetted_study.bed',
             'subsetted_study.fam',
             'subsetted_study.bim',
             'subsetted_study.nosex',
             'subsetted_study.log'])
    
    ##add one new added functino for subsetted_study
    #remove multiallelic
    subsetted_study_remove_multiallelic_file = 'subsetted_study_remove_multiallelic'
    multi_allelic_ids_excluded(bim_file='subsetted_study.bim',
                               var_ids_file='multiallelic.study.in')
    if args.delete_intermediate_files:
        files_to_delete.append('multiallelic.study.in')
    remove_multiallelic_study_completed_process = subset_variants(
        file_to_subset=subsetted_study_file,
        subsetted_file=subsetted_study_remove_multiallelic_file,
        variants_to_keep='multiallelic.study.in',
        path_to_plink=args.plink_path,
        verbose=args.verbose)
    if args.delete_intermediate_files:
        files_to_delete.extend(
            ['subsetted_study_remove_multiallelic.bed',
             'subsetted_study_remove_multiallelic.fam',
             'subsetted_study_remove_multiallelic.bim',
             'subsetted_study_remove_multiallelic.nosex',
             'subsetted_study_remove_multiallelic.log'])

    # Now we have subsetted reference and study files
    # that contain the same variants. Merge them.
    merged_file = 'merged'
    merge_completed_process = five_pass_merge(
        first_file=subsetted_study_remove_multiallelic_file,
        second_file=subsetted_reference_remove_ambiguous_file,
        merged_file=merged_file,
        path_to_plink=args.plink_path,
        verbose=args.verbose)
    if args.delete_intermediate_files:
        files_to_delete.extend(
            ['merged.bed', 'merged.bim',
             'merged.fam', 'merged.log',
             'merged.nosex',
             'merged-merge.missnp',
             'subsetted_reference_remove_ambiguous.bed~',
             'subsetted_reference_remove_ambiguous.bim~',
             'subsetted_reference_remove_ambiguous.fam~',
             'subsetted_study.bed~',
             'subsetted_study.bim~',
             'subsetted_study.fam~'])
    
    # Now run PCA on the merged file:
    #pca_completed_process = run_pca(plink_file=merged_file,
    #                                path_to_plink=args.plink_path,
    #                                verbose=args.verbose)
    pca_completed_process = run_pcair(plink_file=merged_file,
                                    verbose=args.verbose)
    #if args.delete_intermediate_files:
    #    files_to_delete.extend(
    #        ['plink.log', 'plink.nosex',
    #         'plink.eigenvec', 'plink.eigenval',
    #         'plink.eigenvec.var'])

    populations_file = args.reference_populations
    #pcs_df = read_pcs_and_populations(pcs_file='plink.eigenvec',
    #                                  populations_file=populations_file)
    pcs_df = read_pcair_and_populations(pcs_file='PCAIR_output.txt',
                                      populations_file=populations_file)

    study_df, reference_df = split_pcs(pcs_df)

    # Optimize a classifier on the reference data using CV.
    verbose = args.verbose
    optimal_n_pcs, optimal_n_neighbors, list_for_pcs, list_for_neighbors, list_for_accuracy = optimize_classifier(df=reference_df,
                                                                                                                  max_n_pcs=args.number_of_principal_components,
                                                                                                                  max_n_neighbors=args.number_of_neighbors,
                                                                                                                  verbose=verbose)
    # Plot the landscape figure
    arrary_pcs=np.array(list_for_pcs)
    arrary_neighbors=np.array(list_for_neighbors)
    arrary_accuracy=np.array(list_for_accuracy)

    arrary_pcs_2d=np.reshape(arrary_pcs,(args.number_of_principal_components-1,args.number_of_neighbors//5))
    arrary_neighbors_2d=np.reshape(arrary_neighbors,(args.number_of_principal_components-1,args.number_of_neighbors//5))
    arrary_accuracy_2d=np.reshape(arrary_accuracy,(args.number_of_principal_components-1,args.number_of_neighbors//5))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(arrary_pcs_2d, arrary_neighbors_2d, arrary_accuracy_2d, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Number of principal components')
    ax.set_ylabel('Number of neighbors')
    ax.set_zlabel('Accuracy')
    plt.savefig('landscape.png')

    # Fit the optimized classifier and use it generate predictions.
    ref_X = reference_df.drop(['IID', 'Population', 'Ancestry'], axis=1).iloc[:, 0:optimal_n_pcs]
    ref_y = reference_df['Ancestry']
    knc = KNeighborsClassifier(n_neighbors=optimal_n_neighbors)
    knc.fit(ref_X, ref_y)

    study_X = study_df.drop(['IID', 'Population', 'Ancestry'], axis=1).iloc[:, 0:optimal_n_pcs]
    study_y_pred_series = Series(knc.predict(study_X), index=study_df['IID'])

    study_y_pred_proba = knc.predict_proba(study_X)
    study_y_pred_proba_df = pd.DataFrame(study_y_pred_proba,
                                         columns=knc.classes_,
                                         index=study_df['IID'])

    study_y_pred_proba_max = np.max(study_y_pred_proba, axis=1)
    # Make output table.
    output_df = pd.DataFrame(study_y_pred_proba,
                             columns=knc.classes_,
                             index=study_df['IID'])
    output_df.insert(0, 'predicted_ancestry', study_y_pred_series)
    output_df.insert(1, 'MAX', study_y_pred_proba_max)

    output_df.to_csv(args.output_file)
    print(f"Finished generating {args.output_file}.")

    if args.delete_intermediate_files:
        paths_to_delete = [Path(f) for f in files_to_delete]
        [f.unlink() for f in paths_to_delete if f.is_file()]


if __name__ == "__main__":
    main()
