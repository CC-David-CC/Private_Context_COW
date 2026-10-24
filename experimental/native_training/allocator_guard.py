"""Add an opt-in bounded allocator entry to the private pinned GGML copy.

The upstream size-only scheduler entry resets explicit placements and splits the
graph. Measuring there and splitting it again can change placement/copy state.
This separate entry measures and allocates the same prepared splits, retaining
the caller's assignments. Existing GGML entry points are unchanged.
"""
from pathlib import Path
import hashlib

DECLARATION = '''    // PCC private training extension: measure and bound the same assigned splits.
    GGML_API bool ggml_backend_sched_alloc_graph_bounded(ggml_backend_sched_t sched,
        struct ggml_cgraph * graph, const size_t * limits, size_t * sizes);
'''

DEFINITION = '''// PCC private training extension; the ordinary allocation entry is unchanged.
bool ggml_backend_sched_alloc_graph_bounded(ggml_backend_sched_t sched,
        struct ggml_cgraph * graph, const size_t * limits, size_t * sizes) {
    GGML_ASSERT(sched && limits && sizes);
    GGML_ASSERT((int) sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);
    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;
    // Do not reset explicit tensor assignments or split the graph twice.
    ggml_backend_sched_split_graph(sched, graph);
    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph,
        sched->node_backend_ids, sched->leaf_backend_ids, sizes);
    for (int i = 0; i < sched->n_backends; ++i) {
        if (sizes[i] > limits[i]) return false;
    }
    // Size-only reservation records offsets but does not create buffers.
    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph,
            sched->node_backend_ids, sched->leaf_backend_ids)) return false;
    if (!ggml_backend_sched_alloc_splits(sched)) return false;
    for (int i = 0; i < sched->n_backends; ++i) {
        if (ggml_gallocr_get_buffer_size(sched->galloc, i) > limits[i]) return false;
    }
    sched->is_alloc = true;
    return true;
}

'''


def apply(dependency):
    dependency = Path(dependency)
    header = dependency/'include/ggml-backend.h'
    source = dependency/'src/ggml-backend.cpp'
    declaration_anchor = '    GGML_API bool                 ggml_backend_sched_alloc_graph('
    definition_anchor = 'bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {'
    text = header.read_text()
    if DECLARATION not in text:
        assert text.count(declaration_anchor) == 1, 'Unrecognized pinned scheduler header'
        header.write_text(text.replace(declaration_anchor, DECLARATION+declaration_anchor), newline='\n')
    text = source.read_text()
    if DEFINITION not in text:
        assert text.count(definition_anchor) == 1, 'Unrecognized pinned scheduler implementation'
        source.write_text(text.replace(definition_anchor, DEFINITION+definition_anchor), newline='\n')
    return dict(header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
