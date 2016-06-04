import itertools, multiprocessing, logging, os, collections, random, math, sys, time, scipy, scipy.sparse, igraph
from itertools import groupby, combinations
from operator import *
from scipy.stats import hypergeom
import numpy as np, pandas as pd

# from sample_pos_and_neg import *
# from utilities import *
# from configure import *

def load_table_as_list(f, delimiter='\t', encoding=None):    
    if encoding is None:
        f = open(f)
    else:
        import codecs
        f = codecs.open(f, encoding=encoding)
    tmp = [x.strip().split(delimiter) for x in f.read().splitlines()]
    f.close()
    return tmp    


def make_tree(graph, method='priority', edge_name='smallest_parent', parent_priority=None, edge_priority=None, default_priority=None, optim='max'):
    """Returns copy of graph with new edge attribute marking spanning tree"""

    graph = graph.copy()

    if method=='priority':

        assert 1 == (parent_priority is not None) + (edge_priority is not None)            
        if edge_priority is not None: assert default_priority is not None

        if optim=='min': optim=min
        if optim=='max': optim=max

        graph.es[edge_name] = False

        for v in graph.vs:
            parents = graph.neighbors(v.index, mode='out')

            if len(parents) > 0:

                """Choose the parent with the highest valued priority"""
                if parent_priority is not None:
                    small_parent = optim(parents, key=lambda p: parent_priority[p])
                elif edge_priority is not None:
                    small_parent = optim(parents, key=lambda p: edge_priority.get(graph.get_eid(v.index, p), default_priority))

                graph.es[graph.get_eid(v.index, small_parent)][edge_name] = True
            
    else:
        raise Exception('Method not supported')

    return graph

def get_ancestor_matrix(graph, node_order, verbose=False):
    """
    Compute common ancestor matrix.

    lca[a,b] = index of least common ancestor of terms a and b
    """        

    print 'graph size:', len(graph.vs)

    if verbose: time_print('Calculating connectivity matrix')
    d = np.int8(np.isfinite(np.array(graph.shortest_paths(graph.vs, graph.vs, mode='out'), order='C')))

    ancestor_matrix = np.zeros(d.shape, dtype=np.int32)
    ancestor_matrix.fill(-1)

    if verbose: time_print('Iterating:')
    for idx, i in enumerate(node_order):

        # Note: includes self as a child
        children_list = np.where(d[:,i] == 1)[0]
        
        # For those descendants without a computed LCA yet, set their LCA to this term
        lca_sub = ancestor_matrix[children_list.reshape(-1,1), children_list]
        lca_sub[lca_sub == -1] = i
        ancestor_matrix[children_list.reshape(-1,1), children_list] = lca_sub

    # Check symmetry
    assert (ancestor_matrix.T == ancestor_matrix).all()
    assert (-1 == ancestor_matrix).sum() == 0, 'The ontology may have more than one root'

    return ancestor_matrix

def get_smallest_ancestor(graph, node_sizes, method='ancestor_matrix', verbose=False):
    # Node order: smallest to largest terms
    return get_ancestor_matrix(graph,
                               [a[0] for a in sorted(enumerate(node_sizes), key=lambda a: a[1])],
                               verbose)

def get_lca_matrix(graph):
    ## I might deprecate this. Consider using get_smallest_ancestor instead
    return get_ancestor_matrix(graph, graph.topological_sorting(mode='out'))

# Minimum 

def collapse_node(g, v, edge_filter=None, use_v_name=False, combine_attrs=None, default_attr=None, verbose=True):

    if use_v_name:
        assert isinstance(v, str)
        v = g.vs.find(name_eq=v).index

    if verbose: print 'Inside:', g.vs[v]['term_description']

    in_edges = g.es[g.incident(v, mode='in')]
    out_edges = g.es[g.incident(v, mode='out')]

    if edge_filter is not None:
        in_edges = [e for e in in_edges if edge_filter(e)]
        out_edges = [e for e in out_edges if edge_filter(e)]

    for e_in in in_edges:

        for e_out in out_edges:

            in_neigh, out_neigh = e_in.source, e_out.target

            # Only add an edge if it doesn't already exist
            if g[in_neigh, out_neigh] == 0:
                g.add_edge(in_neigh, out_neigh)
                e = g.es[g.get_eid(in_neigh, out_neigh)]
                if combine_attrs is not None:
                    # Set default value of edge attributes to 0
                    for key in combine_attrs:  e[key] = None

            e = g.es[g.get_eid(in_neigh, out_neigh)]

            # Update attributes
            if combine_attrs is not None:
                for key in combine_attrs:
                    e[key] = combine_attrs[key](e_in, e_out, e)
                    if verbose and key=='triangle_edge_priority':
                        print 'Setting', key, g.vs[in_neigh]['name'], g.vs[out_neigh]['name'], 'to', combine_attrs[key](e_in, e_out, e), (e_in[key], e_out[key])

            e['collapsed_length'] = e_in['collapsed_length'] + e_out['collapsed_length']
            e['collapsed_terms'] = e_in['collapsed_terms'] + [g.vs[v]['name']] + e_out['collapsed_terms']

    g.delete_vertices(v)    
    return g

def map_gene_pairs_2_triangles(ontology, gsets):
    """Assign each gene pair to it's smallest common ancestor.  This is
    it's 'triangle'.  It also works out that this is both of their lowest
    terms, such that this is like a within-term interaction"""

    graph_orig = ontology.get_igraph()
    graph = ontology.get_igraph().copy()
    graph.add_vertices(ontology.genes)
    assert graph.vs[-len(ontology.genes):]['name'] == ontology.genes
    assert graph.vs[:len(ontology.terms)]['name'] == ontology.terms
    graph.add_edges([(g, t) for g, terms in [(len(ontology.terms) + ontology.genes_index[g], terms) for g, terms in ontology.gene_2_terms.items()] for t in terms])
    gene_sca_matrix = get_smallest_ancestor(graph, ontology.get_term_sizes().tolist() + [1 for i in ontology.genes], verbose=True)
    term_2_genes = {ontology.terms_index[t] : set(g) for t, g in ontology.get_term_2_genes().items()}

    # tmp = [(i, j, gene_sca_matrix[len(ontology.terms) + i, len(ontology.terms) + j], graph_orig.neighbors(gene_sca_matrix[len(ontology.terms) + i, len(ontology.terms) + j], mode='in')) for i, j in [(ontology.genes_index[g1], ontology.genes_index[g2]) for g1, g2 in gsets]]
    # triangles = [(sca, [t for t in terms if i in term_2_genes[t]], [t for t in terms if j in term_2_genes[t]]) for i, j, sca, terms in tmp]

    #--- Alternate triangle calculation.  Allow for multiple SCA in anti-chain paths
    # Make sure the order is 'C'.  Set the diagonal to True (even though it should technically be False) for easier use of np.all below
    connectivity = ontology.get_connectivity_matrix()
    not_connectivity = np.logical_not(np.array(connectivity, dtype=np.bool, order='C'))
    np.fill_diagonal(not_connectivity, True)

    children_dict = {v.index : np.array(graph_orig.neighbors(v.index, mode='in')) for v in graph_orig.vs}
    gene_2_terms_set = {g : np.array(terms) for g, terms in ontology.gene_2_terms.items()}
    term_2_genes = {ontology.terms_index[t] : set(g) for t, g in ontology.get_term_2_genes().items()}

    time_print('Calculating triangles')
    triangles = ((g1, g2, gene_2_terms_set[g1], gene_2_terms_set[g2]) for g1, g2 in gsets)
    triangles = ((g1, g2, g1_terms, g2_terms, np.intersect1d(g1_terms, g2_terms, assume_unique=True)) for g1, g2, g1_terms, g2_terms in triangles)
    # This could be potentially double-counting a term pair if for a single gene pair, they have more than one sca
    # triangles = [[(sca, np.intersect1d(g1_terms, children_dict[sca], assume_unique=True), np.intersect1d(g2_terms, children_dict[sca], assume_unique=True)) for sca in terms[not_connectivity[terms, :][:, terms].all(axis=0)]] for g1, g2, g1_terms, g2_terms, terms in triangles]
    triangles = [[(sca,
                   tuple(np.intersect1d(g1_terms, children_dict[sca], assume_unique=True)),
                   tuple(np.intersect1d(g2_terms, children_dict[sca], assume_unique=True))) \
                      for sca in terms[not_connectivity[terms, :][:, terms].all(axis=0)]] \
                 for g1, g2, g1_terms, g2_terms, terms in triangles]

    ## Each element corresponds to a gene pair.  Each element is a
    ## list of triplets (LCA, children terms spanned by gene 1,
    ## children terms spanned by gene 2)
    return triangles

def count_siblings(triangles, num_terms, term_pair_triangles=None):
    """Returns the number of times that a sibling occurs in a set of triangles"""

    triu_indices = np.triu_indices(num_terms, k=1)

    if term_pair_triangles is None:
        term_pair_triangles = [set([ (t1, t2) if t1 < t2 else (t2, t1) for sca, side1, side2 in path for t1 in side1 for t2 in side2]) \
                               for path in triangles]

    return scipy.sparse.coo_matrix(([1 for term_pairs in term_pair_triangles for t1, t2 in term_pairs],
                                    ([t1 for term_pairs in term_pair_triangles for t1, t2 in term_pairs],
                                     ([t2 for term_pairs in term_pair_triangles for t1, t2 in term_pairs]))),
                                   shape=(num_terms, num_terms)).toarray()[triu_indices]


def siblings_2_gene_pairs(triangles):
    """Returns dictionary mapping a sibling (t1, t2), where the term
indices are t1<t2, to the indices of gene pairs.  This is the same as
indices to elements in <triangles>"""
    
    return {a : [y for x, y in b] for a, b in groupby(sorted([((t1, t2) if t1<t2 else (t2, t1), i) for i, tri in enumerate(triangles) for \
                                                              sca, t1_list, t2_list in tri for t1 in t1_list for t2 in t2_list],
                                                             key=lambda a:a[0]), key=lambda a:a[0])}
def lcas_2_gene_pairs(triangles):
    """Returns dictionary mapping a term to the indices of gene pairs.
This is the same as indices to elements in <triangles> """

    return {a : [y for x, y in b] for a, b in groupby(sorted([(sca, i) for i, tri in enumerate(triangles) for \
                                                                sca, t1_list, t2_list in tri],
                                                               key=lambda a:a[0]), key=lambda a:a[0])}

def explain_gene_pairs(triangles, lcas, siblings):
    """Given a list of enriched siblings and lcas, returns those siblings
and lcas that a gene pair falls under.
    
    For each element in triangles, returns a pair (x, y) where x are
    terms in lcas and y are term pairs in siblings.
    """

    lcas, siblings = set(lcas), set([(t1,t2) if t1<t2 else (t2, t1) for t1, t2 in siblings])
    return [(set([sca for sca, t1_list, t2_list in tri]) & lcas,
             set([(t1,t2) if t1<t2 else (t2,t1) for sca, t1_list, t2_list in tri for t1 in t1_list for t2 in t2_list])) \
            for tri in triangles]
    
class Ontology:
    
    def __init__(self,
                 ontology_file,
                 mapping_file=None,
                 combined_file=False,
                 shuffle_genes=False,
                 parent_child=False,
                 genes_as_terms=False,
                 encoding=None,
                 add_root=None,
                 verbose=True):
        """
        ontology: File path to term-to-term edges
        mapping:  File path to gene-to-term edges        
        """

        if isinstance(ontology_file, str) and (not os.path.exists(ontology_file)) and ontology_file in ['all', 'bp', 'cc', 'mf']:
            ontology_file = os.path.join(os.getenv('HOME'), 'GI/NNstructures/start_mkramer-collapsed/evidence_no_IGI/propagate_forward/ontology_%s/precollapse.propagated' % ontology_file)
            if verbose: print 'Assuming ontology file is', ontology_file
            # else:
            #     raise Exception("%s isn't recognized as an ontology" % ontology_file)

        ontology_table = load_table_as_list(ontology_file, encoding=encoding) if (isinstance(ontology_file, str) and os.path.exists(ontology_file)) else ontology_file
            
        if combined_file:
            mapping_table = [x for x in ontology_table if x[2]=='gene']
            ontology_table = [x for x in ontology_table if x[2]!='gene']
        else:
            if mapping_file is None:                
                mapping_file = ontology_file + '.mapping'      
                if verbose: print 'Assuming that mapping file is', mapping_file
            mapping_table = load_table_as_list(mapping_file, encoding=encoding) if (isinstance(mapping_file, str) and os.path.exists(mapping_file)) else mapping_file

        if parent_child:
            ontology_table = [[x[1],x[0]]+list(x[2:]) for x in ontology_table]
            mapping_table = [[x[1],x[0]]+list(x[2:]) for x in mapping_table]
            
        if genes_as_terms:
            # Turn each gene into a term, and represent the genes with
            # dummy copies. E.g. if gene g is annotated to term t, then
            # make g into a term, and then create a copy g_dummy that
            # is annotated to g and is now the "gene"
            ontology_table = ontology_table + mapping_table
            mapping_table = [[u'dummy_'+g, g, 'gene'] for g in set([x[0] for x in mapping_table])]

        ## Read term-to-term edges        
        # term_2_terms[<term_name>] --> list of <term_name>'s children terms
        term_2_terms = {r: tuple(p[0] for p in q) for r, q in \
                           itertools.groupby(sorted(ontology_table,
                                                    key=lambda a:a[1]),
                                             key=lambda a:a[1])}

        # Extract parent-child relations, if they exist
        if all([len(x)==3 for x in ontology_table]):
            # relation_dict[(child, parent)] --> edge relation, e.g. 'is_a' or 'part_of'            
            self.relation_dict = {(a,b) : relation for a, b, relation in ontology_table}


        if add_root is not None:
            ## Check if there is a single unifying root term of the ontology. If not, then identify the multiple roots and join them under an artificial root
            root_list = set(term_2_terms.keys()) - set([y for x in term_2_terms.values() for y in x])
            if len(root_list) > 1:
                root_name = add_root
                term_2_term[root_name] = root_list
                if self.has_attr('relation_dict'):
                    for r in root_list:
                        relation_dict[(r, root_name)] = 'artificial_root'
                

        ## Read term-to-term edges        
        # child_2_parents[<term_name>] --> list of <term_name>'s parent term names
        child_2_parents = {r: tuple(p[1] for p in q) for r, q in \
                           itertools.groupby(sorted(ontology_table,
                                                    key=lambda a:a[0]),
                                             key=lambda a:a[0])}

        ## Read gene-to-term edges
        # gene_2_terms[<gene_name>] --> list of terms that <gene_name> is mapped to
        gene_2_terms = {key: tuple(set([a[1] for a in group])) for key, group in \
                            itertools.groupby(sorted(mapping_table,
                                                     key=lambda a:a[0]),
                                              key=lambda a:a[0])}

        if shuffle_genes:            

            # Shuffle all annotated genes in the ontology
            if isinstance(shuffle_genes, bool):
                shuffle_genes = list(gene_2_terms.keys())
            # Only shuffle a subset of genes
            elif isinstance(shuffle_genes, collections.Iterable):
                shuffle_genes = list(set(gene_2_terms.keys()) & set(shuffle_genes))
            else:
                raise Exception('Unsupported argument for shuffle_genes')

            if verbose: time_print('Shuffling %s genes' % len(shuffle_genes))

            gene_permutation = range(len(shuffle_genes))
            random.shuffle(gene_permutation)
            gene_2_terms = {shuffle_genes[gene_permutation[i]] : gene_2_terms[shuffle_genes[i]] for i in range(len(shuffle_genes))}

        ## Check that the set of terms is the same according to term_2_terms and gene_2_terms
        terms_A = set.union(set(term_2_terms.keys()), *[set(x) for x in term_2_terms.values()])
        terms_B = set.union(*[set(x) for x in gene_2_terms.values()])

        # print len(terms_A), len(terms_B), len(terms_A & terms_B)
        # assert terms_B.issubset(terms_A)

        if not terms_B.issubset(terms_A):
            if verbose: print 'WARNING: There are {} orphaned terms that are annotated to genes but not connected to the rest of the ontology'.format(len(terms_B - terms_A))
            # print 'WARNING: Removing orphaned terms'
            # gene_2_terms = {g : sorted(set(t) & terms_A) for g, t in gene_2_terms.items()}        
            
        terms = sorted(list(terms_A | terms_B))
#        terms = sorted(list(terms_A))
        genes = sorted(gene_2_terms.keys())
        
        ## terms_index[<term_name>] --> index in self.terms
        terms_index = {b:a for a,b in enumerate(terms)}

        ## genes_index[<gene_name>] --> index in self.genes        
        genes_index = {b:a for a,b in enumerate(genes)}

        ## Convert gene_2_terms to list term indices rather than term names
        for k, v in gene_2_terms.items():
            gene_2_terms[k] = [terms_index[x] for x in gene_2_terms[k]]

        ## Read term-to-term edges        
        # child_2_parents_indices[<term_name>] --> list of indices of <term_name>'s parent terms
        child_2_parents_indices = {r : [terms_index[x] for x in v] for r, v in child_2_parents.items()}


        self.term_2_terms = term_2_terms
        self.child_2_parents = child_2_parents
        self.child_2_parents_indices = child_2_parents_indices
        self.gene_2_terms = gene_2_terms
        self.terms = terms
        self.terms_index = terms_index
        self.genes = genes
        self.genes_index = genes_index
        
        if verbose: print 'Done constructing ontology'

    def collapse_ontology(self, subsume_child):
        #(g, v, edge_filter=None, use_v_name=False, combine_attrs=None, default_attr=None, verbose=True):

        g = self.get_igraph()

        self.propagate_annotations()

        # Get term_2_genes
        del self.term_2_genes
        term_2_genes = {t : set(g_list) for t, g_list in self.get_term_2_genes().items()}

        # Iterate from leaves up
        for v in g.topological_sorting(mode='out'):
            v_genes = term_2_genes[g.vs[v]['name']]
            if len(v_genes)==0:
                g = collapse_node(g, v, verbose=False)
            if any([v_genes==term_2_genes[g.vs[p]['name']] for p in g.neighbors(v, mode='out')]):
                g = collapse_node(g, v, verbose=False)

        
    def delete_terms(self, terms_to_delete):
        print 'Returning new ontology with %s terms removed' % len(terms_to_delete)
        from tempfile import NamedTemporaryFile
        ontology_file = NamedTemporaryFile()
        ontology_file.write(
            '\n'.join([t1+'\t'+t2+'\t'+self.relation_dict[(t2,t1)] for t1,t2_list in self.term_2_terms.items() for t2 in t2_list if (t1 not in terms_to_delete and t2 not in terms_to_delete)]) + '\n' +
            '\n'.join([g + '\t' + self.terms[t] + '\tgene' for g, t_list in self.gene_2_terms.items() for t in t_list if self.terms[t] not in terms_to_delete]) + '\n')
        ontology_file.flush()
        return Ontology(ontology_file.name, combined_file=True, parent_child=False)

    # ## It's too difficult to mutate the ontology structure. It'd be better to just remake the ontology using the constructor
    # def delete_terms(self, terms_to_remove, remove_orphaned_genes=True):
    #     print 'Deleting %s terms' % len(terms_to_remove)

    #     term_2_genes = self.get_term_2_genes()

    #     for t in terms_to_remove:
    #         t_index = self.terms_index[t]

    #         self.terms.remove(t)

    #         try: del self.term_2_terms[t]
    #         except: pass
                
    #         try: del self.child_2_parents[t]
    #         except: pass
            
    #         try: del self.child_2_parents_indices[t_index]
    #         except: pass

    #         #print t, type(t), [self.genes[g] for g in term_2_genes[t]]
    #         for g in term_2_genes[t]:
    #             # print 'gene/term', self.genes[g], t
    #             # print self.gene_2_terms[self.genes[g]]
    #             self.gene_2_terms[self.genes[g]].remove(t_index)               

    #     self.terms_index = {b : a for a, b in enumerate(self.terms)}
    #     for g_orphan in [g for g, x in self.gene_2_terms.items() if len(x)==0]:
    #         self.genes.remove(g_orphan)
    #     self.genes_index = {b : a for a, b in enumerate(self.genes)}

    #     del self.term_2_genes



    # def write_ontology(self, out):
    #     out.write(


    def transitive_closure(self, g=None):

        relations = ['is_a', 'regulates', 'positively_regulates', 'negatively_regulates', 'has_part', 'part_of', 'gene']
        relations_index = {b : a for a, b in enumerate(relations)}
        go_reasoning = dict()
        for r in relations:
            go_reasoning[('is_a', r)] = r
            go_reasoning[(r, 'is_a')] = r
            if r != 'has_part':
                go_reasoning[(r, 'part_of')] = 'part_of'
        for r in ['regulates', 'positively_regulates', 'negatively_regulates']:
            go_reasoning[(r, 'part_of')] = 'regulates'
        go_reasoning[('has_part', 'has_part')] = 'has_part'

        if g is None:
            g = self.get_igraph()

        # Sort from leaves up
        for i in g.topological_sorting(mode='out'):
        
            # Get parents
            for j in g.neighbors(i, mode='out'):
                
                # Infer GO relations of new edges
                base_relation = g.es[g.get_eid(i, j)]['relation']

                # Iterate over grandparents
                for p in g.neighbors(j, mode='out'):
                    r = go_reasoning.get((base_relation, g.es[g.get_eid(j, p)]['relation']), None)

                    # If a relation can't be inferred, then don't add a new edge
                    if r is not None:
                        if -1 != g.get_eid(i, p, error=False):
                            # Edge already exists, so take the higher-ranked relation
                            e = g.es[g.get_eid(i, p)]
                            e['relation'] = r if relations_index[r] > relations_index[e['relation']] else e['relation']
                        else:
                            # Add new edge with relation
                            g.add_edge(i, p, relation=r)                                        

        return g
        
        ## Update term_2_terms , child_2_parents, child_2_parents_indices

    # def a(
    #     d = self.get_connectivity_matrix()

    #     x[~ d[x.reshape(-1,1), x].any(1)]

    def semantic_similarity(self, genes_subset=None, term_sizes='subset', between_terms=False, output='Resnik'):
        """Computes the semantic similarity between pair of genes in
        <genes_subset>. Similarity s(g1,g2) is defined as
        -log_2(|T_sca| / |T_root|) where |T| is the number of genes in
        <genes_subset> that are under term T. T_sca is the "smallest
        common ancestor", the common ancestral term with the smallest
        term size. T_root is the root term of the ontology.

        genes_subset : The set of genes, over which pairs the
        similarity will be calculated. If <term_sizes>='subset', then
        term sizes will be recalculated according to only these genes,
        rather than all genes in the ontology

        between_terms : if True, then output similarity between all
        terms

        output: type of semantic similarity
        """

        # If no genes are specified, then compute the similarity between all pairs of genes
        if genes_subset is None: genes_subset = self.genes

#        genes_subset = genes_subset[:5]

        time_print('Calculating term sizes')
        if term_sizes=='all_genes':
            term_sizes = np.array(self.get_term_sizes())
        elif term_sizes=='subset':
            # Recompute term sizes, with respect to the intersection of genes
            term_2_genes = self.get_term_2_genes()
            genes_subset_set = set(genes_subset)
            term_sizes = np.array([len(set([self.genes[x] for x in term_2_genes[t]]) & genes_subset_set) for t in self.terms])
        else:
            raise Exception()

        if output=='Resnik':
            graph = self.get_igraph().copy()

            sca = get_smallest_ancestor(graph, term_sizes)
            ss_terms = -1 * np.log2(term_sizes[sca] / float(term_sizes.max()))

            if between_terms:
                return ss_terms
            else:
                # For convenience below, set the similarity between a term and itself to be 0
                ss_terms[[np.arange(ss_terms.shape[0]), np.arange(ss_terms.shape[0])]] = 0                
                
                idx_list = [self.gene_2_terms[g] for g in genes_subset]
                ss = [ss_terms[idx1, :][:, idx2].max() for idx1, idx2 in combinations(idx_list, 2)]
                return ss

            # # Add nodes in the igraph object to represent genes
            # graph.add_vertices(self.genes)
            # graph.add_edges([(g, t) for g, t_list in self.gene_2_terms.items() for t in t_list])
            # assert graph.vs[-len(self.genes):]['name'] == self.genes

            # sca = get_smallest_ancestor(graph, term_sizes)
            # # Offset the indices
            # idx = [len(self.terms) + self.genes_index[g] for g in genes_subset]
            # ss = (-1 * np.log2(term_sizes / float(term_sizes.max())))[sca[idx, :][:, idx][np.triu_indices(len(idx), k=1)]]

            # return ss
        elif output=='sca_list':
            ## For each pair of gene, return a list of the smallest
            ## common ancestors (sca). There may be more than one sca with the same size.

            gene_2_terms_numpy = {g : np.array(t_list) for g, t_list in self.gene_2_terms.items()}            
            common_ancestors = [np.intersect1d(gene_2_terms_numpy[g1], gene_2_terms_numpy[g2], assume_unique=True) \
                                for g1, g2 in combinations(genes_subset, 2)]
            assert all(x.size > 0 for x in common_ancestors)
            min_size = [term_sizes[x].min() for x in common_ancestors]
            sca_list = [x[term_sizes[x]==m] for x, m in zip(common_ancestors, min_size)]

            # Dict: (g1,g2) gene pairs --> list of term indices
            return {(g1,g2) : x for (g1, g2), x in zip(combinations(genes_subset, 2), sca_list)}

    def get_term_2_genes(self, verbose=True):
        if not hasattr(self, 'term_2_genes'):
            if verbose: print 'Calculating term_2_genes'
            self.term_2_genes = {self.terms[c]: [self.genes_index[x[0]] for x in d] \
                                     for c, d in itertools.groupby(sorted([(a,t) for a, terms in self.gene_2_terms.items() for t in terms],
                                                                        key=lambda x:x[1]),
                                                                   key=lambda x:x[1])}
            for t in self.terms:
                if not self.term_2_genes.has_key(t): self.term_2_genes[t] = []

        return self.term_2_genes

    def get_term_sizes(self):
        "Returns an array of term sizes in the same order as self.terms"

        if not hasattr(self, 'term_sizes'):
            self.term_sizes = np.vstack([np.bincount(x, minlength=len(self.terms)) for x in self.gene_2_terms.values()]).sum(0)
#            self.term_sizes = np.bincount(np.concatenate([np.array(x) for x in self.gene_2_terms.values()]))
#            self.term_sizes = np.bincount(reduce(lambda a,b : a+b, self.gene_2_terms.values()))
        
        return self.term_sizes

    def get_term_descriptions(self,f=os.path.join(os.getenv('HOME'),'GI/data/GO/filtered-GO/goID_2_name.tab'), quote=':'):

        if not hasattr(self, 'term_description'):
            self.term_description = {x.split('\t')[0].replace(quote, '_'):x.split('\t')[1] for x in open(f).read().splitlines()}
            for t in self.terms:
                if not self.term_description.has_key(t): self.term_description[t] = 'NA'

        return self.term_description

    def get_term_namespace(self,f=os.path.join(os.getenv('HOME'),'GI/data/GO/filtered-GO/goID_2_namespace.tab'), quote=':'):
        if not hasattr(self, 'term_namespace'):
            self.term_namespace = {x.split('\t')[0].replace(quote, '_'):x.split('\t')[1] for x in open(f).read().splitlines()}
            for t in self.terms:
                if not self.term_namespace.has_key(t): self.term_namespace[t] = 'NA'

        return self.term_namespace

    # def get_leaves(self):
    #     # Find leaves
    #     # Probably to debug
    #     leaves = [t for t in terms if not self.term_2_terms.has_key(t)]
    #     return(leaves)

    def restrict_genes(self, subset):
        
        assert set(subset).issubset(set(self.gene_2_terms.keys()))

        self.genes = sorted(subset)
        self.gene_2_terms = {g : self.gene_2_terms[g] for g in self.genes}
        self.genes_index = {b:a for a,b in enumerate(self.genes)}

    def subset_terms(self, subset_type, threshold=None, size=None):
        """Returns the indices of a subset of terms"""

        assert (subset_type not in ['gt', 'gte', 'lt', 'lte', 'eq']) or (threshold is not None)
        assert (subset_type not in ['random']) or (size is not None)

        if subset_type=='gt':
            term_indices = [i for i, x in enumerate(self.get_term_sizes()) if x > threshold]
        if subset_type=='gte':
            term_indices = [i for i, x in enumerate(self.get_term_sizes()) if x >= threshold]
        if subset_type=='lt':
            term_indices = [i for i, x in enumerate(self.get_term_sizes()) if x < threshold]
        if subset_type=='lte':
            term_indices = [i for i, x in enumerate(self.get_term_sizes()) if x <= threshold]
        if subset_type=='eq':
            term_indices = [i for i, x in enumerate(self.get_term_sizes()) if x == threshold]
        if subset_type=='random':
            term_indices = random.sample(range(len(self.terms)), size)

        return term_indices

    def shuffle(self):
        pass

        ## OLD CODE: PROBABLY BUGGY

        # # Convert gene_2_terms to term_2_genes
        # logger.debug('Converting gene_2_terms to term_2_genes')
        # genes_index = {b:a for a,b in enumerate(mapped_genes)}
        # term_2_genes = {a:[genes_index[q[0]] for q in b] for a, b in
        #                 itertools.groupby(sorted([(g, t) for g in
        #                                           gene_2_terms.keys() for t in gene_2_terms[g]],
        #                                          key=lambda x:x[1]), key=lambda x: x[1])}
        # for t in range(len(terms)):
        #     if not term_2_genes.has_key(t): term_2_genes[t] = []

        # if 'shuffle_gene_degree' in shuffle_options:
        #     # For each gene separately, shuffle the terms that it is annotated to
        #     logger.debug('Shuffling annotations while preserving gene degree')
        #     indices = range(len(terms))
        #     gene_2_terms = {k: tuple(random.sample(indices, len(v))) for k, v in gene_2_terms.items()}

        # elif 'shuffle_term_degree' in shuffle_options:
        #     # For each term separately, shuffle the genes that it is annotated to
        #     # -- Only shuffle to other genes that have at least one annotation            
        #     logger.debug('Shuffling annotations while preserving term degree')
        #     genes_index = {b:a for a,b in enumerate(mapped_genes)}
        #     indices = range(len(mapped_genes))
        #     term_2_genes = {key: tuple(random.sample(indices, len(set([genes_index[a[0]] for a in group])))) for key, group in \
        #                     itertools.groupby(mapping, key=lambda a:a[1])}

        #     # Convert term_2_genes to gene_2_terms
        #     mapping = [(mapped_genes[g], k) for k, v in term_2_genes.items() for g in v]
        #     gene_2_terms = {key: tuple(set([terms_index[a[1]] for a in group])) for key, group in \
        #                     itertools.groupby(mapping, key=lambda a:a[0])}

        # if 'shuffle_gene_names' in shuffle_options:
        #     logger.debug('Shuffling gene names in gene-term relations')
        #     shuffled_genes = mapped_genes[:]
        #     random.shuffle(shuffled_genes)
        #     gene_permutation = dict(zip(mapped_genes, shuffled_genes))
        #     mapping = [(gene_permutation[x], y) for x,y in mapping]
        # else: logger.debug('Not shuffling gene names in gene-term relations')

        # if 'shuffle_all_degree' in shuffle_options:
        #     gene_column = [x[0] for x in mapping]
        #     term_column = [x[1] for x in mapping]
        #     random.shuffle(term_column)
        #     mapping = zip(gene_column, term_column)

    @classmethod
    def get_ann_count(cls, ann_defn):
        ## OLD CODE : PROBABLY BUGGY

        return {x.split('\t')[0] : int(x.split('\t')[1]) for x in open(ann_defn).read().splitlines()}
        
        # if os.path.isfile(ann_defn):
        #     logger.debug('Reading annotation frequencies from %s', ann_defn)
        #     ann_count = {x.split('\t')[0] : int(x.split('\t')[1]) for x in open(ann_defn).read().splitlines()}
        #     ann_count = {key: (ann_count[key] if ann_count.has_key(key) else 0) for key in genes}       

        # elif ann_defn=='precollapse':
        #     logger.debug('Using precollapse annotation frequencies')

        #     # Annotation count of genes before collapsing and before propagation
        #     precollapse_ann_count = {x.split('\t')[0] : int(x.split('\t')[1]) for x in
        #                              open(os.path.join(os.path.dirname(ontology_prefix), 'precollapse.annotation_count')).read().splitlines()}
        #     # Restrict keys (because <precollapse_ann_count> may
        #     # describe more genes than <genes>) and fill in keys in
        #     # <genes> not in <precollapse_ann_count>
        #     precollapse_ann_count = {key: (precollapse_ann_count[key] if precollapse_ann_count.has_key(key) else 0) for key in genes}               

        #     # Check that genes are annotated to 0 terms in
        #     # precollapse_ann_count (pre-collapse) if and only if they are
        #     # annotated to 0 terms in gene_2_terms (post-collapse)
        #     # -- Note: this won't hold true if terms are restricted to leaves
        #     if (not use_leaves) and len(set(['shuffle_term_degree']) & set(shuffle_options))==0:
        #         assert set(g for g, k in gene_2_terms.items() if len(k)==0) == set(g for g,k in precollapse_ann_count.items() if k==0)

        #     # Check that dictionary is over same set of genes
        #     assert set(precollapse_ann_count.keys()) == set(genes)

        #     ann_count = precollapse_ann_count
        # else:
        #     raise Exception('Unhandled ann_defn: %s' % ann_defn)
            
    @classmethod
    def get_ontology_prefix(cls, **kwargs):
        ## OLD!
        ontology_prefix = os.path.join(kwargs['NNstructures'],
                                       'start_%s' % kwargs['start_ontology'],
                                       'evidence_%s' % kwargs['evidence'],
                                       'propagate_%s' % kwargs['propagate'],
                                       'ontology_%s' % kwargs['ontology'],
                                       '%s.propagated' % kwargs['collapse_scheme'])
        
        return ontology_prefix

    @classmethod
    def create_ontology(cls, folder, start_ontology, evidence, ontology, collapse_scheme, genes=None):
        ## IRRELVANT TO SAL
        ontology = Ontology(Ontology.get_ontology_prefix(NNstructures=folder,
                                                         start_ontology=start_ontology,
                                                         evidence=evidence,
                                                         propagate='forward',
                                                         ontology=ontology,
                                                         collapse_scheme=collapse_scheme),
                            genes,
                            'precollapse',
                            ontology,
                            False,
                            shuffle_options=[])
        
        return ontology

    @classmethod
    def to_remove(self, gset_list, ann_thresh, logger=None):
        ## Apply annotation threshold

        assert isinstance(ann_thresh, int)

        if ann_thresh > 0:
            to_remove = [any(self.ann_count[g] < ann_thresh for g in gset) for gset in gset_list]
        else:
            to_remove = [False for gset in gset_list]

        logger.debug('Applying annotation threshold of %s: removing %s (out of %s) genes and %s (out of %s) gsets',
                     ann_thresh,
                     sum(x < ann_thresh for x in self.ann_count.values()),
                     len(self.ann_count.keys()),
                     sum(to_remove),
                     len(to_remove))

        return to_remove

    def get_igraph(self):
        """The vertices in the resulting graph will be in the same order as
        self.terms
        """

        self.graph = igraph.Graph(n=len(self.terms),
                                  edges=[(self.terms_index[c], self.terms_index[p]) for p, children in self.term_2_terms.items() for c in children],
                                  directed=True,
                                  vertex_attrs={'name':self.terms})
        
        # if not hasattr(self, 'graph'):

        #     time_print('Reading into igraph')
        #     import igraph
        #     graph = igraph.Graph.Read_Ncol(self.ontology_file)

        #     time_print('Permuting vertices')
        #     # Permute vertices of graph to be in the same order as self.terms
        #     from code.utilities import permute_x_to_y
        #     graph = graph.permute_vertices(permute_x_to_y(self.terms, graph.vs['name']))
        #     assert(graph.vs['name'] == self.terms)

        #     ontology_table = load_table_as_list(self.ontology_file)
        #     if all([len(x)==3 for x in ontology_table]):
        #         # Add edge relations, if they exist
        #         graph.es[graph.get_eids([(a,b) for a, b, r in ontology_table])]['relation'] = [r for a, b, r in ontology_table]            
            
        #     time_print('Done reading into igraph')

        #     self.graph = graph
        
        return self.graph
        
    def get_common_ancestors(self, x, y):
        """
        Returns list of indices of the common ancestors between terms with index a and b.

        Ancestors are sorted by term size.
        """
        
        d = self.get_connectivity_matrix()

        return sorted(np.intersect1d(d[x,:].nonzero()[0], d[y,:].nonzero()[0], assume_unique=True),
                      key=lambda a: self.term_sizes[a])

    def get_lca_matrix(self):
        """
        Compute least common ancestor matrix.

        lca[a,b] = index of least common ancestor of terms a and b
        """        

        d = self.get_connectivity_matrix()

        time_print('Constructing LCA matrix')

        lca_matrix = np.zeros(d.shape, dtype=np.int32)
        lca_matrix.fill(-1)

        # Topological sorting, going bottom-up the tree
        for i in self.get_igraph().topological_sorting(mode='out'):

            # Note: includes self as a child
            children_list = np.where(d[:,i] == 1)[0]

            # For those descendants without a computed LCA yet, set their LCA to this term
            lca_sub = lca_matrix[children_list.reshape(-1,1), children_list]
            lca_sub[lca_sub == -1] = i
            lca_matrix[children_list.reshape(-1,1), children_list] = lca_sub

        # Check symmetry
        assert (lca_matrix.T == lca_matrix).all()
        assert (-1 == lca_matrix).sum() == 0

        time_print('Done constructing LCA matrix')

        self.lca_matrix = lca_matrix
        return self.lca_matrix

    def get_smallest_ancestor(self):
        
        d = self.get_connectivity_matrix()

        time_print('Constructing smallest ancestor matrix')

        ancestor_matrix = np.zeros(d.shape, dtype=np.int32)
        ancestor_matrix.fill(-1)

        #for i in self.graph.topological_sorting(mode='out')

        # Traverse nodes, from smallest to largest terms
        for i in map(lambda a: a[0], sorted(enumerate(self.get_term_sizes()), key=lambda a: a[1])):
            
            # Note: includes self as a child
            children_list = np.where(d[:,i] == 1)[0]

            # For those descendants without a computed LCA yet, set their LCA to this term
            lca_sub = ancestor_matrix[children_list.reshape(-1,1), children_list]
            lca_sub[lca_sub == -1] = i
            ancestor_matrix[children_list.reshape(-1,1), children_list] = lca_sub

        # Check symmetry
        assert (ancestor_matrix.T == ancestor_matrix).all()
        assert (-1 == ancestor_matrix).sum() == 0

        time_print('Done constructing ancestor matrix')

        self.ancestor_matrix = ancestor_matrix
        return self.ancestor_matrix

    def get_shortest_paths(self, sparse=False, chunk_size=500):
        """
        d[a,b] = length of the shortest path from a to b, bottom-up the ontology
        """
        
        graph = self.get_igraph()

        if sparse:
#            i, j, lengths = zip(*[(i, v) for i, vpath in enumerate(graph.get_shortest_paths(graph.vs, graph.vs, mode='out', output='vpath')) for v in vpath])
#            return scipy.sparse.vstack([scipy.sparse.csr_matrix(graph.shortest_paths(v, graph.vs, mode='out')) for v in graph.vs])
            print len(graph.vs), chunk_size
            return scipy.sparse.vstack([scipy.sparse.csr_matrix(graph.shortest_paths(graph.vs[x[0]:x[1]], graph.vs, mode='out')) \
                                        for x in split_indices_chunk(len(graph.vs), chunk_size)])
        else:
            return np.array(graph.shortest_paths(graph.vs, graph.vs, mode='out'), order='C')
#        return np.array(graph.shortest_paths(graph.vs, graph.vs, mode='out'), order='C')
        
    def get_longest_paths(self):
        """
        d[a,b] = length of the longest path from a to b, bottom-up the ontology
        """

        graph = self.get_igraph()

        return -1 * np.array(graph.shortest_paths(graph.vs, graph.vs, weights=[-1 for x in graph.es], mode='out'), order='C')

    def get_connectivity_matrix(self, sparse=False):
        """
        Creates a term-by-term matrix d where d[a,b] is 1 if term b is
        an ancestor of term a, and 0 otherwise.

        Note:
           d[a,a] == 1
           d[root,j] == 0, for every j

        """

        if not hasattr(self, 'd'):
            time_print('Calculating connectivity matrix')
            if sparse:
                paths = self.get_shortest_paths(sparse=True)
                self.d = scipy.sparse.coo_matrix((np.isfinite(paths[paths.nonzero()]),
                                                  (paths.nonzero()[0], paths.nonzero()[1])),
                                                 dtype=np.int8)
            else:
                self.d = np.int8(np.isfinite(self.get_shortest_paths()))

        return self.d

    def get_connectivity_matrix_nodiag(self):
        """
        Returns a similar matrix as in Ontology.get_connectivity_matrix(),
        but the diagonal of the matrix is 0.

        Note: !!!!!!!!!!!!!!!!!!!!!!!!
            d[a, a] == 0 instead of 1

        """

        if not hasattr(self, 'd_nodiag'):
            d = self.get_connectivity_matrix()
            self.d_nodiag = d.copy()
            self.d_nodiag[np.diag_indices(self.d_nodiag.shape[0])] = 0
            assert not np.isfortran(self.d_nodiag)

        return self.d_nodiag

    def get_leaves(self, terms_list, children_list=None):
        """
        Select the terms in <terms_list> that are not ancestors of any term in <children_list>.
        
        If <children_list> is None, then select the terms in <terms_list> that are not ancestors of any of the other terms in <terms_list>.

        """
        
        connectivity_matrix_nodiag = self.get_connectivity_matrix_nodiag()
        
        terms_list = np.array(terms_list)
        if children_list is None:
            children_list = terms_list
        else:
            children_list = np.array(children_list)

        return terms_list[~ np.any(connectivity_matrix_nodiag[children_list, :][:, terms_list], axis=0)]

    def propagate_annotations(self, direction='forward', method='iterative_union'):
        """                                                                                                                                                                                                        
        Propagates the gene-term annotations through the ontology                                                                                                                                                  
        """

        if direction=='forward':
            if method=='iterative_union':

                child_2_parents_idx = {self.terms_index[c] : [self.terms_index[p] for p in p_list] for c, p_list in self.child_2_parents.items()}
                gene_2_terms_set = {g : set(t_list) for g, t_list in self.gene_2_terms.items()}

                genes_to_update = self.gene_2_terms.keys()
                count = 0
                while len(genes_to_update) > 0:
                    # Iterate over a copy of genes_to_update
                    for g in genes_to_update[:]:
                        curr_terms = gene_2_terms_set[g]
                        new_terms = set([p for t in curr_terms for p in child_2_parents_idx.get(t, [])])

                        union_terms = curr_terms | new_terms
                        if len(union_terms) == len(curr_terms):
                            genes_to_update.remove(g)
                        else:
                            gene_2_terms_set[g] = union_terms
                
                    print count,
                    count +=1

                    if count == 1000:
                        0 / asdf 
                print
                self.gene_2_terms = {g : sorted(t_set) for g, t_set in gene_2_terms_set.items()}
            else:
                ancestor_matrix = np.array(self.get_connectivity_matrix(), dtype=np.int32)
                self.gene_2_terms = {g : ancestor_matrix[t, :].sum(0).nonzero()[0].tolist() for g, t in self.gene_2_terms.items()}

            if hasattr(self, 'term_2_genes'):
                del self.term_2_genes
            if hasattr(self, 'term_sizes'):
                del self.term_sizes

        else:
            raise Exception('Unsupported direction')

        if hasattr(self, 'connectivity_matrix'):
            del self.connectivity_matrix


    # def propagate_annotations(self, direction='forward', sparse=False):
    #     """
    #     Propagates the gene-term annotations through the ontology
    #     """

    #     if direction=='forward':
    #         if sparse:
    #             ancestor_matrix = scipy.sparse.csr_matrix(self.get_connectivity_matrix(sparse=sparse), dtype=np.int32)
    #         else:
    #             ancestor_matrix = np.array(self.get_connectivity_matrix(sparse=sparse), dtype=np.int32)

    #         self.gene_2_terms = {g : ancestor_matrix[t, :].sum(0).nonzero()[0].tolist() for g, t in self.gene_2_terms.items()}
    #         if hasattr(self, 'term_2_genes'):
    #             del self.term_2_genes
    #         if hasattr(self, 'term_sizes'):
    #             del self.term_sizes

    #     else:
    #         raise Exception('Unsupported direction')

    def propagate(self, ontotypes, prop, ontotype_size, max_ontotype, method='fixed_size'):
        """
        Propagates a list of base ontotypes.
        
        ontotypes: An array of ontotypes in scipy.sparse.csr_matrix format.  Each row is a separate ontotype.

        method: If 'fixed_size', then the sum of values in every ontotype is exactly the same, namely <ontotype_size>
        
        ontotype_size: If method=='fixed_size', then this is the sum of values in every ontotype
        """


        # Just return the array

        if prop=='genes':

            if method=='fixed_size':

                if ontotype_size == 0:
                    return ontotypes

                assert ontotype_size in [1,2]        

                ## Each non-zero term in the ontotype is assumed to be hit
                ## by exactly one gene and that term is the only term the
                ## gene hits
                graph = self.get_igraph()

                time_print('Calculating node ancestry')
                d = self.get_connectivity_matrix()

                assert d.dtype == ontotypes.dtype

                # The usage of np.int8 for <d> requires ontotype size be <= 127
                assert ontotype_size <= 127

                time_print('Formatting ontotype array')
                ontotype_arr = scipy.sparse.csr_matrix(ontotypes)
                nnz = ontotype_arr.nonzero()

                time_print('Sanity checks')

                # # Number of starting perturbations in each ontotype
                # # Currently supporting only 2
                # assert ontotype_size==2

                # Check that the number of non-zero elements in every row is equal ontotype_size
                assert( ((ontotype_arr != 0).sum(axis=1) != ontotype_size).sum() == 0 )

                # Check that the every consecutive block of <ontotype_size> elements is located in a new row a new row
                assert( (nnz[0][0:len(nnz[0]):ontotype_size,] != range(ontotype_arr.shape[0])).sum() == 0 )

                # Check that the non-zero elements were returned in the order of traversing row-by-row
                assert( (np.argsort(np.int64(nnz[0]) * ontotype_arr.shape[1] + np.int64(nnz[1])) != np.array(range(len(nnz[0])))).sum() == 0)

                # Check that the number of non-zero elements is a multiple of ontotype_size
                assert( len(nnz[1]) % ontotype_size == 0 )

                # import pdb
                # pdb.set_trace()

                time_print('Doing propagation')
                # For all ontotypes, calculate the propagation of the
                # first term perturbed as an array, and likewise for the
                # second term.  Then, add the arrays.

                if ontotype_size == 2:
                    # Matrix for propagating the first perturbed term
                    a = np.take(d, nnz[1][0:len(nnz[1]):2], axis=0) * \
                        np.reshape(np.array(ontotype_arr[(nnz[0][0:len(nnz[0]):2],
                                                          nnz[1][0:len(nnz[1]):2])]), (len(nnz[0])/2, 1))

                    # Matrix for propagating the second perturbed term
                    b = np.take(d, nnz[1][1:len(nnz[1]):2], axis=0) * \
                        np.reshape(np.array(ontotype_arr[(nnz[0][1:len(nnz[0]):2],
                                                          nnz[1][1:len(nnz[1]):2])]), (len(nnz[0])/2, 1))

                    # Since the we're propagating genes, we need to limit the ontotype value
                    ontotype_arr = np.minimum(a + b, max_ontotype)

                elif ontotype_size == 1:
                    ontotype_arr = np.take(d, nnz[1], axis=0) * \
                        np.reshape(np.array(ontotype_arr[(nnz[0], nnz[1])]), (len(nnz[0]), 1))

                    assert (ontotype_arr > max_ontotype).sum().sum() == 0
                    ontotype_arr = np.minimum(ontotype_arr, max_ontotype)

                time_print('Done')
                return ontotype_arr

            elif method=='topological':

                print 'IN:', 'GO_0000001' in self.term_2_terms.keys()
                print self.term_2_terms.keys()[:5]

                assert ontotypes.dtype == np.int8
                if ontotypes.dtype == np.int8:
                    assert max_ontotype <= 127
                
                if scipy.sparse.issparse(ontotypes):
                    ontotypes = ontotypes.toarray(order='F')
                else:
                    ontotypes = np.array(ontotypes, order='F')

                # Topological sort on terms, going bottom-up the ontology
                for i in self.get_igraph().topological_sorting(mode='out'):
                    
                    # Add the values from child to all its parents.
                    # Cap the value at <max_ontotype>.  When
                    # propagating the gene-term annotations in an
                    # ontology, you should set max_ontotype=1.
                    if self.terms[i] in self.child_2_parents_indices.keys():

                        parents = np.array(self.child_2_parents_indices[self.terms[i]])

                        ontotypes[:, parents] = np.minimum(max_ontotype,
                                                           ontotypes[:, parents] + ontotypes[:, i].reshape(ontotypes.shape[0], 1))

                        if not np.all(ontotypes <= max_ontotype):
                            print i
                            assert False
                assert np.all(ontotypes <= max_ontotype)

                return ontotypes

    def get_features(self, gset_sample, prop, format='dict', dtype=np.int8, gene_ordering=None):
        """ Computes the ontotype for each gset in <gset_sample>
        according to the propagation scheme <prop>.  Returns a list,
        where each element is a dictionary representation of an
        ontotype.
        
        gset_sample = list of gene sets (i.e. gsets)
        A "gset" is a set of genes, represented as a tuple

        prop == 'genes' means the value of a term is the number of
        genes in the term which are deleted.

        scipy.coo : The Scipy Sparse COO format http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix

        When format=='scipy.coo', this is how you convert it to a
        dense matrix in a text file.  Let the return matrix be x.  Run
        np.savetxt(<filename.txt>, x.toarray(), sep='\t').  Look up
        toarray() function in Scipy Sparse doc page and the np.savetxt
        function.
        """
        
        if prop=='genes':

            time_print('Creating features')
            print format, format=='scipy.csr'
            
            if format=='dict':
                from collections import Counter                
                onto_sample = [reduce(add, [self.gene_2_terms[g] for g in gset]) for gset in gset_sample]                
                onto_sample = [dict(Counter(idxs)) for idxs in onto_sample]
                
            elif format=='scipy.coo':
                time_print('Making i, j, data')

                gene_2_terms = {k: np.array(v) for k, v in self.gene_2_terms.items()}
                i = np.repeat(np.arange(len(gset_sample)), [sum(gene_2_terms[g].size for g in gset) for gset in gset_sample])
                j = np.concatenate([gene_2_terms[g] for gset in gset_sample for g in gset])

                #ij = np.array([(a, t) for a, gset in enumerate(gset_sample) for g in gset for t in self.gene_2_terms[g]])
                #data = np.ones((ij.shape[0], ))
                data = np.ones((i.size, ))
                time_print('Making sparse COO matrix')
                #onto_sample = scipy.sparse.coo_matrix((data, (ij[:,0], ij[:,1])), (len(gset_sample), len(self.terms)), dtype=dtype)
                onto_sample = scipy.sparse.coo_matrix((data, (i, j)), (len(gset_sample), len(self.terms)), dtype=dtype)

            elif format=='scipy.csr':
                time_print('Making indices, indptr, data')
                gene_2_terms = {k: np.array(v) for k, v in self.gene_2_terms.items()}
                gset_sample_x = [np.concatenate([gene_2_terms[g] for g in gset]) if len(gset)>0 else np.array([]) for gset in gset_sample]
                indices = np.concatenate(gset_sample_x)
                indptr = np.concatenate((np.array([0]),
                                         np.cumsum([gset.size for gset in gset_sample_x])))
                data = np.ones((len(indices), ), dtype=dtype)

                time_print('Making sparse CSR matrix')
                onto_sample = scipy.sparse.csr_matrix((data, indices, indptr), (len(gset_sample), len(self.terms)), dtype=dtype)

            elif format=='dense':
                dim = len(gset_sample[0])
                assert dim <= 127, "Gset dimension must be <= 127 to create an np.int8 array"

                # Check that all genes in gset_sample are in gset_2_terms
                assert set.union(*[set(x) for x in gset_sample]).issubset(set(self.gene_2_terms.keys()))

                # Convert gene_2_terms to a numpy genes-by-terms boolean matrix
                time_print('Creating gene_2_terms matrix')
                gene_2_terms_arr = np.zeros((len(self.genes), len(self.terms)), dtype=np.int8)
                for g, terms in self.gene_2_terms.items():
                    gene_2_terms_arr[self.genes_index[g], terms] = 1

                time_print('Converting gset_sample into matrix')
                gset_sample_arr = np.array([[self.genes_index[g] for g in gset] for gset in gset_sample])

                time_print('Creating gset_2_terms matrix')
                onto_sample = np.zeros((len(gset_sample), len(self.terms)), dtype=np.int8)
                for d in range(dim):
                    onto_sample += np.take(gene_2_terms_arr,
                                           gset_sample_arr[:,d],
                                           axis=0)

            time_print('Done creating features')

            return onto_sample

        elif prop=='or':

            onto_sample = self.get_features(gset_sample, prop='genes', format=format)

            if format=='dict':
                # Set all values to 1
                onto_sample = [{k: 1 for k, v in d.items()} for d in onto_sample]
            elif format in ['scipy.coo', 'scipy.csr', 'dense']:
                # Do this to preserve data type of onto_sample,
                # instead of onto_sample = (onto_sample >= 1), which
                # would create a boolean
                onto_sample[ onto_sample >= 1 ] = 1
            else:
                raise Exception('Format %s not supported for prop' % (format, prop))

            return onto_sample

        elif prop=='children':

            import igraph, numpy, collections

            graph = igraph.Graph.Read_Ncol(self.ontology_prefix)
            g_vs_index = {b:a for a,b in enumerate(graph.vs['name'])}                
            gene_2_terms = self.gene_2_terms
            
            # For each gset, create a dictionary mapping a perturbed term
            # (i.e. its index in self.terms) to the number of genes
            # perturbed
            genes_features = [dict(collections.Counter([term for g in gset for term in gene_2_terms[g]])) for gset in gset_sample]      
            
            for term_count in genes_features:
                parent_perturbations = [self.terms_index[x] for \
                                        x in graph.vs[[p for term in term_count.keys() for \
                                                       p in graph.neighbors(g_vs_index[self.terms[term]], mode='out')]]['name']]
                parent_perturbations = dict(Counter(parent_perturbations))                    
                term_count.update(parent_perturbations)
            perturbations = genes_features

            return perturbations

        elif prop=='min_cut':
            # TODO
            raise Exception('Not supported')

        elif prop=='gene_identity':
            tmp = [(i, self.genes_index[g]) for i, gset in enumerate(gset_sample) for g in gset]
            i, j = zip(*tmp)            
            data = np.ones(len(i), dtype=dtype)
            onto_sample = scipy.sparse.coo_matrix((data, (np.array(i), np.array(j))), (len(gset_sample), len(self.genes)))

            if format=='scipy.csr':
                onto_sample = scipy.sparse.csr_matrix(onto_sample)

            assert format in ['scipy.coo', 'scipy.csr']

            return onto_sample

        elif prop=='matrix_mult':
            assert isinstance(gset_sample, (np.ndarray, pd.DataFrame)) or scipy.sparse.issparse(gset_sample), 'gset_sample must be a sample-by-gene matrix'

            if isinstance(gset_sample, pd.DataFrame):
                gene_ordering = gset_sample.columns
                gset_sample = np.array(gset_sample)

            gset_sample = scipy.sparse.coo_matrix(gset_sample)
            annotation_matrix = self.get_annotation_matrix()

            if gene_ordering is not None:
                contained = np.array([self.genes_index.has_key(g) for g in gene_ordering])
                gset_sample = scipy.sparse.coo_matrix(scipy.sparse.csc_matrix(gset_sample)[:,contained])
                subset = np.array([self.genes_index[g] for g in gene_ordering if self.genes_index.has_key(g)])
                annotation_matrix = scipy.sparse.csc_matrix(scipy.sparse.csr_matrix(annotation_matrix)[subset,:])

            onto_sample = gset_sample * annotation_matrix

            if format==pd.DataFrame:
                # Create dataframe
                unused_terms = (onto_sample!=0).toarray().sum(0) == 0
                print 'Removing %s terms with no mutations among samples' % unused_terms.sum()
                onto_sample = pd.DataFrame(onto_sample.toarray()[:,~unused_terms], columns=np.array(self.terms)[~unused_terms])

            return onto_sample

        else:
            raise Exception('Invalid perturbation type')

    def get_annotation_matrix(self):

        # Convert gene names to indices
        gene_2_terms = [(self.genes_index[g], t_list) for g, t_list in self.gene_2_terms.items()]

        self.annotation_matrix = scipy.sparse.coo_matrix(([1 for g, t_list in gene_2_terms for t in t_list],
                                                          ([g for g, t_list in gene_2_terms for t in t_list],
                                                           [t for g, t_list in gene_2_terms for t in t_list])),
                                                         shape=(len(self.genes), len(self.terms)))
        return self.annotation_matrix

    def format_features(self, features, format='mike_format'):

        if format=='mike_format':
            
            return '\n'.join(['\t'.join([str(x)+'\t'+str(y) for x,y in sorted(z.items(), key=lambda a:a[0])]) for z in features])

        elif format in ['scipy.coo', 'scipy.csr', 'numpy.dense']:

            time_print('Getting element indices for sparse representation')
            i = [ind for ind, feat in enumerate(features) for x in range(len(feat))]
            tmp = [y for feat in features for x in feat.items() for y in x]
            j = tmp[0:len(tmp):2]
            data = tmp[1:len(tmp):2]
            
            time_print('Creating sparse COO matrix')
            features = scipy.sparse.coo_matrix((data, (i, j)), shape=(len(features), len(self.terms)), dtype=np.int8)

            time_print('Converting to specified format')
            if format=='scipy.csr':
                features = scipy.sparse.csr_matrix(features)
            elif format=='numpy.dense':
                features = features.todense()
            
            return features        



    def pair_enrichment(self, labels, logger=None):

        if logger is None: logger = logging.getLogger(multiprocessing.current_process().name)

        logger.debug('Calculating pairwise term enrichment')

        logger.debug('Calculating background likelihood')
        gsets = [(x, y) for x, y, z in zip(labels.labels, labels.Gset.gset_list, labels.Gset.to_remove) if not z]        
        total_pos = sum(x for x, y in gsets)
        total_neg = len(gsets) - total_pos        
        background_ll = float(total_pos) / (total_pos + total_neg)
        logger.debug('Total Pos / Neg: %s / %s', total_pos, total_neg)
        logger.debug('Background likelihood: %s', background_ll)
                                                
        assert labels.Gset.dims == [2]

        logger.debug('Forming term pairs')
        term_pairs = list(itertools.combinations(range(len(self.terms)), 2))
        term_pairs_index = {b:a for a,b in enumerate(term_pairs)}

        pos_gsets = [gset for pos, gset in gsets if pos]
        neg_gsets = [gset for pos, gset in gsets if not pos]

        # g = igraph.Graph.Read_Ncol(self.ontology_prefix)
        # g_vs_index = {b:a for a,b in enumerate(g.vs['name'])}        
        # m = numpy.matrix(g.get_adjacency().data)

        # root = np.where(m.sum(1) == 0)[0][0,0]

        
        # part = dict()
        # part[root] = pos_gset

        # # Calculate gset_2_terms

        # for v in g.bfs(root, mode='in')[0]:
        #     current_gsets = part[v]

        #     children = g.neighbors(v, mode='in')
        #     for child in children:
        #         part[child] = [gset for gset in current_gsets if child in gset_2_terms[gset]]                            
        #     for child1, child2 in itertools.combinations(g.vs[children]['name'], 2):
        #         term_pairs_index[terms_index[child1], terms_index[child2]
        #         terms_index[child2]
        
        #     # All pairs of terms
            
        #     del part[v]

        gene_2_terms_set = { g: set(t) for g, t in self.gene_2_terms.items() }

        n = len(self.terms)

        #def f(i, j, n): return n * i - i * (i+1) / 2 + j- i - 1

        p = multiprocessing.Pool(processes=16)

        
        # pos_gsets = pos_gsets[:10000]
        # neg_gsets = neg_gsets[:10000]

        # pos_gsets = random.sample(pos_gsets, min(len(pos_gsets), 300000))
        # neg_gsets = random.sample(neg_gsets, min(len(neg_gsets), 300000))

        logger.debug('Calculating spanning pos gene pairs')
        logger.debug('Time: %s', time.time())
        #pos_spanning = [term_pairs_index[(tA, tB) if tA < tB else (tB, tA)] for g1, g2 in pos_gsets for tA in self.gene_2_terms[g1] for tB in self.gene_2_terms[g2] if tA != tB]

        rem, quot = len(pos_gsets) % 16, len(pos_gsets) / 16
        pos_gsets_groups = [pos_gsets[i * quot + (i if i < rem else rem):
                                      (i+1) * quot + (i+1 if i < rem else rem)] for i in range(16)]

        # pos_spanning_union = p.map(span_union, [(x, gene_2_terms_set, n) for x in pos_gsets_groups])
        pos_spanning_symm = p.map(span_11, [(x, gene_2_terms_set, n) for x in pos_gsets_groups])

        # pos_spanning_union = [b for a in p.map(span_union, [(x, gene_2_terms_set, n) for x in pos_gsets_groups]) for b in a]
        # pos_spanning_symm = [b for a in p.map(span_11, [(x, gene_2_terms_set, n) for x in pos_gsets_groups]) for b in a]

        # pos_spanning_union = [n*b[0] - b[0]*(b[0]+1)/2 + b[1] - b[0] - 1 if b[0] < b[1] else n*b[1] - b[1]*(b[1]+1)/2 + b[0] - b[1] - 1 \
        #                 for g1, g2 in pos_gsets for b in itertools.combinations(gene_2_terms_set[g1] | gene_2_terms_set[g2], 2)]
        # pos_spanning_symm = [n*b[0] - b[0]*(b[0]+1)/2 + b[1] - b[0] - 1 if b[0] < b[1] else n*b[1] - b[1]*(b[1]+1)/2 + b[0] - b[1] - 1 \
        #                 for g1, g2 in pos_gsets for b in itertools.combinations(gene_2_terms_set[g1] ^ gene_2_terms_set[g2], 2)]

        logger.debug('Calculating spanning neg gene pairs')
        logger.debug('Time: %s', time.time())
        #neg_spanning = [term_pairs_index[(tA, tB) if tA < tB else (tB, tA)] for g1, g2 in neg_gsets for tA in self.gene_2_terms[g1] for tB in self.gene_2_terms[g2] if tA != tB]

        rem, quot = len(neg_gsets) % 16, len(neg_gsets) / 16
        neg_gsets_groups = [neg_gsets[i * quot + (i if i < rem else rem):
                                      (i+1) * quot + (i+1 if i < rem else rem)] for i in range(16)]

        # neg_spanning_union = p.map(span_union, [(x, gene_2_terms_set, n) for x in neg_gsets_groups])
        neg_spanning_symm = p.map(span_11, [(x, gene_2_terms_set, n) for x in neg_gsets_groups])

        # neg_spanning_union = [b for a in p.map(span_union, [(x, gene_2_terms_set, n) for x in neg_gsets_groups]) for b in a]
        # neg_spanning_symm = [b for a in p.map(span_11, [(x, gene_2_terms_set, n) for x in neg_gsets_groups]) for b in a]

        # neg_spanning_union = [n*b[0] - b[0]*(b[0]+1)/2 + b[1] - b[0] - 1 if b[0] < b[1] else n*b[1] - b[1]*(b[1]+1)/2 + b[0] - b[1] - 1 \
        #                 for g1, g2 in neg_gsets for b in itertools.combinations(gene_2_terms_set[g1] | gene_2_terms_set[g2], 2)]
        # neg_spanning_symm = [n*b[0] - b[0]*(b[0]+1)/2 + b[1] - b[0] - 1 if b[0] < b[1] else n*b[1] - b[1]*(b[1]+1)/2 + b[0] - b[1] - 1 \
        #                 for g1, g2 in neg_gsets for b in itertools.combinations(gene_2_terms_set[g1] ^ gene_2_terms_set[g2], 2)]

        logger.debug('Summing up counts')
        logger.debug('Time: %s', time.time())
        ## densities = [[0, len(self.term_2_genes[tA]) * len(self.term_2_genes[tB])] for tA, tB in term_pairs]
        densities = [[0, 0] for tA, tB in term_pairs]
        for span in pos_spanning_symm:
            for pair_idx, count in span.iteritems():
                densities[pair_idx][0] += count
        for span in neg_spanning_symm:
            for pair_idx, count in span.iteritems():
                densities[pair_idx][1] += count

        # for pair_idx, count in collections.Counter(pos_spanning_union).iteritems():
        #     densities[pair_idx][0] = count
        # for pair_idx, count in collections.Counter(neg_spanning_union).iteritems():
        #     densities[pair_idx][1] = count

        logger.debug('Calculating likelihood enrichment')
        logger.debug('Time: %s', time.time())
        # Likelihood enrichment
        ll_enrichment = [ round((float(a) / (a+b)) / background_ll, 3) if (a+b) != 0 else 0.0 for a, b in densities ]
        
        # # from scipy.stats import hypergeom
        # logger.debug('Calculating hypergeometric enrichment')
        # logger.debug('Start %s', time.time())
        
        # p = multiprocessing.Pool(processes=16)
        # # densities_groups = [densities[i * int(math.ceil(len(densities) / 16.)) : min(len(densities), (i+1) * int(math.ceil(len(densities) / 16.)))] for i in range(16)]

        # densities = densities[:100]

        # rem, quot = len(densities) % 16, len(densities) / 16
        # densities_groups = [densities[i * quot + (i if i < rem else rem):
        #                               (i+1) * quot + (i+1 if i < rem else rem)] for i in range(16)]
        # hyper_enrichment = [b for a in p.map(calc_hyper_enrich, [(total_pos, total_neg, x) for x in densities_groups]) for b in a]
        # logger.debug('Finished calculating hypergeometric enrichment')
        # logger.debug('End %s', time.time())

        #hyper_enrichment = [1 - hypergeom.cdf(a, total_pos + total_neg, total_pos, b) for a, b in densities]
    
        # self.term_2_genes = term_2_genes

        logger.debug('Finished Time: %s', time.time())

        return ll_enrichment, densities

    def single_enrichment(self, labels, logger=None):

        if logger is None: logger = logging.getLogger(multiprocessing.current_process().name)

        logger.debug('Calculating single term enrichment')

        logger.debug('Calculating background likelihood')
        gsets = [(x, y) for x, y, z in zip(labels.labels, labels.Gset.gset_list, labels.Gset.to_remove) if not z]        
        total_pos = sum(x for x, y in gsets)
        total_neg = len(gsets) - total_pos        
                                                
        assert labels.Gset.dims == [2]

        gene_2_terms_set = {g: set(t) for g, t in self.gene_2_terms.items()}

        logger.debug('Calculating spanning pos gene pairs')
        pos_gsets = [gset for pos, gset in gsets if pos]
        pos_spanning = [b for g1, g2 in pos_gsets for b in gene_2_terms_set[g1] & gene_2_terms_set[g2]]
        logger.debug('Calculating spanning neg gene pairs')
        neg_gsets = [gset for pos, gset in gsets if not pos]
        neg_spanning = [b for g1, g2 in neg_gsets for b in gene_2_terms_set[g1] & gene_2_terms_set[g2]]

        logger.debug('Summing up counts')
        #densities = [[0, len(self.terms_2_genes[t]) * (len(self.terms_2_genes[t]) - 1) / 2] for t in range(len(self.terms))]
        densities = [[0, 0] for t in range(len(self.terms))]
        for term_idx, count in collections.Counter(pos_spanning).iteritems():
            densities[term_idx][0] = count
        for term_idx, count in collections.Counter(neg_spanning).iteritems():
            densities[term_idx][1] = count
    
        logger.debug('Calculating likelihood enrichment')
        # Likelihood enrichment
        background_ll = float(total_pos) / (total_pos + total_neg)
        logger.debug('Total Pos / Neg: %s / %s', total_pos, total_neg)
        logger.debug('Background likelihood: %s', background_ll)
        ll_enrichment = [ round((float(a) / (a+b)) / background_ll, 3) if (a+b) != 0 else 0.0 for a, b in densities ]

        return ll_enrichment, densities


class OntologyGraph(igraph.Graph):
    def __init__(self, n=None, edges=None, directed=None, graph_attrs=None, vertex_attrs=None, edge_attrs=None):
        igraph.Graph.__init__(self, n=None, edges=None, directed=None, graph_attrs=None, vertex_attrs=None, edge_attrs=None)



def write_ontology_structure():
    pass
def write_ontology_annotations():
    pass

def simulate_ontology(Gset, mapping, orig_int):

    term_2_genes = {key: tuple(set([genes_index[a[0]] for a in group])) for key, group in \
                    itertools.groupby(mapping, key=lambda a:a[1])}
    term_sizes = {term: len(genes) for term, genes in term_2_genes.items()}

    ess_win = [0,400]
    ess_list = [t for t in terms if term_sizes[t] >= ess_win[0] and term_size[t] <= ess_win[1]]

    co_ess_win = [0,250]
    co_ess_list = [t for t in terms if term_sizes[t] >= co_ess_win[0] and term_size[t] <= co_ess_win[1]]
    
    interactions = []
    
    # Randomly pick a number of terms that are essential
    while (len(interactions) < len(orig_int)):
        if random.random() < 0.5:
            # Randomly pick a number of terms that are essential
            ess = random.choice(ess_list)
            ess_list.remove(ess)
            # Make interactions between every pair in term
            interactions.extend(itertools.combinations(term_2_genes[ess], 2))
        else:
            # Randomly pick a number of terms that are co-essential
            co_ess_1, co_ess_2 = random.sample(co_ess_list, 2)
            co_ess_list.remove(co_ess_1)
            co_ess_list.remove(co_ess_2)

            interactions.extend([[x,y] for x in term_2_genes[co_ess_1] for y in term_2_genes[co_ess_2]])

    new_labels = [False for i in labels]
    from sample_pos_and_neg import get_gset_idx
    genes = Gset.genes
    for g1, g2 in interactions: new_labels[get_gset_idx[genes[g1], genes[g2]], Gset.gset_idx, Gset.genes_idx] = 1
    return new_labels

    # Set an upper limit of term sizes.  Perhaps co-essential
    # terms. Continue picking terms and co-essential terms until there
    # are at least as many genes in a synthetic lethal relation as
    # there are originally, or as many number of synthetic lethal
    # interactions as there are originally

    # Write annotations and ontology
    

    # Propagate annotations?
    'Rscript --propagate annotations'

    pass

