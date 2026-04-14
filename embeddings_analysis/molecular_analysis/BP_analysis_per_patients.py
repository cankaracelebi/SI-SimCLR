import pandas as pd
from melapa.TableOfPairs import TableOfPairs
from scipy.stats import mannwhitneyu
import argparse
import os
import numpy as np
import yaml

def main():


    FLAGS = argparse.ArgumentParser(description='BP analysis based on PUD per patient.')
    FLAGS.add_argument('--sample_name', help='Sample name')
    FLAGS.add_argument('--config_file', help='Location of config file')

    print("Parsing arguments....")
    args = FLAGS.parse_args()
    with open(args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(" config file has been properly loaded!")


    # df_counts = pd.read_csv(config['df_counts'], index_col=0)
    sample_names = [args.sample_name]
    table_of_pairs = pd.read_csv(config['table_of_pairs'], index_col=0)

    if 'gene_list' in config.keys():
        gene_list = config['gene_list']
    else:
        gene_list = 'Gene_Symbols'
        
    if config['score'] == 'gene':
        cols_sample = [col for col in table_of_pairs.columns if args.sample_name in col and 'gene' not in col][0]
        table_of_pairs[f'gene_{args.sample_name}'] = table_of_pairs[cols_sample]
        table_of_pairs['gene'] = list(table_of_pairs.index)

    annotationGO = pd.read_csv(config['annotationGO'])

    annotationGO[gene_list] = annotationGO[gene_list].str.split(',').apply(lambda x: [i.strip() for i in x])
    annotationGO[gene_list] = annotationGO[gene_list].apply(lambda x: list(set(x)))
    annotationGO = annotationGO[annotationGO[gene_list].apply(lambda x: len(x) > 20)]


    tablePairs = TableOfPairs(df=table_of_pairs, treatments=sample_names, agg_method='mean')

    if 'pathway_ID' in config.keys():
        GO_id = config['pathway_ID']
    else:
        GO_id = 'GO_id'

    if 'pathway_description' in config.keys():
        pathway_description = config['pathway_description']
    else:
        pathway_description = 'Description'

    df_results = tablePairs.inGO_vs_NOTinGO_one_sample_all_terms(df_GO=annotationGO, 
                                                                sample=args.sample_name,
                                                                GO_term_colname=GO_id,
                                                                GO_description_colname=pathway_description,
                                                                gene_list_colname=gene_list,
                                                                per_pair=False,
                                                                score=config['score'],
                                                                bootstrap=True)

    df_results.to_csv(os.path.join(config['path_to_results'], f"{args.sample_name}.csv"))                                        

if __name__ == '__main__':
    main()