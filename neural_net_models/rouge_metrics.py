import rouge

AGGREGATORS = ['Avg', 'Best', 'Individual']


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5f}\t{}: {:5f}\t{}: {:5f}'.format(metric, 'P', p, 'R', r, 'F1', f)


def print_rouge_results(results):
    for line in results:
        print(line)


def get_rouge_results(hypotheses, references, aggregator="Avg"):
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(
        metrics=['rouge-n', 'rouge-l', 'rouge-w'],
        max_n=4,
        limit_length=True,
        length_limit=100,
        length_limit_type='words',
        apply_avg=apply_avg,
        apply_best=apply_best,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True
    )

    scores = evaluator.get_scores(hypotheses, references)
    rouge_results = []

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if aggregator == "Individual":  # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    rouge_results.append('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    rouge_results.append(
                        prepare_results(
                            metric, results_per_ref['p'][reference_id], results_per_ref['r'][reference_id],
                            results_per_ref['f'][reference_id]
                        )
                    )
        else:
            rouge_results.append((prepare_results(metric, results['p'], results['r'], results['f'])))
    return rouge_results


# hypotheses_1 = ["Hello How are you"]
# references_1 = ["Hello How are you uyutf"]
# rogue_results = get_rouge_results(hypotheses_1, references_1)
# print_rouge_results(rogue_results)
