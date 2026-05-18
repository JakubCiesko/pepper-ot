echo "Running evaluation script"
for experiment_id in $(yq ".variants[].overrides.experiment_id" research/configs/experiments/vocab_presentation_effect_config.yaml); do 
	echo "";
	echo "--";
	echo "Evaluating $experiment_id";
	echo "--";
	#PYTHONPATH=. python3 -m research.experiments.cli.main evaluate-run --run research/artifacts/experiments/som_prompting_effect/runs/$experiment_id --gt research/artifacts/human_eval/sgg_gt/eval/human_eval_nonempty_completed.json --gt-only

	PYTHONPATH=. python3 -m research.experiments.cli.main evaluate-run --run research/artifacts/experiments/vocab_presentation_effect/runs/$experiment_id --gt research/artifacts/human_eval/sgg_gt/eval/human_eval_nonempty_completed.json --gt-only
	echo "Evaluation Done for Experiment: $experiment_id";
	echo "--";
	echo;
done



  
