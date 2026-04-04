"""Upload a local motion.npz to the Weights & Biases registry.

Example:
	 python scripts/upload_npz.py --npz_path datasets/motions_npz/run1_subject2.npz --collection_name run1_subject2
"""

from __future__ import annotations

import argparse

import wandb


def main():
	parser = argparse.ArgumentParser(description="Upload motion.npz to W&B registry")
	parser.add_argument("--npz_path", type=str, required=True, help="Path to motion.npz")
	parser.add_argument("--collection_name", type=str, required=True, help="Registry collection name")
	parser.add_argument("--registry_name", type=str, default="motions", help="Registry artifact type (default: motions)")
	parser.add_argument("--project", type=str, default="motions_npz", help="W&B project to log from")
	args = parser.parse_args()

	run = wandb.init(project=args.project, name=args.collection_name)
	artifact = wandb.Artifact(name=args.collection_name, type=args.registry_name)
	# Always store the motion file under a stable name expected by training scripts.
	artifact.add_file(args.npz_path, name="motion.npz")
	logged_artifact = run.log_artifact(artifact)
	run.link_artifact(
		artifact=logged_artifact,
		target_path=f"wandb-registry-{args.registry_name}/{args.collection_name}",
	)
	print(f"[OK] Uploaded: {args.collection_name} -> {args.registry_name}/{args.collection_name}")


if __name__ == "__main__":
	main()
