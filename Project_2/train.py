import jax
import optax
from functools import partial
from loss_function import loss_function
import wandb


@partial(jax.jit, static_argnums=(0, 2))
def train_step(model, params, optimizer, opt_state, batch):
    """Single training step."""

    grad_fn = jax.value_and_grad(loss_function, has_aux=True)
    (loss_value, metrics), grads = grad_fn(params, model, batch)

    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, metrics


def perform_training_steps(
    model,
    params,
    optimizer,
    opt_state,
    num_training_steps,
    batch_size,
    unroll_steps,
    td_steps,
    game_manager,
):
    """Performs a series of training steps."""

    print(f"🏋️‍♀️ Training for {num_training_steps} steps...")
    accumulated_metrics = {
        "total": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "reward": 0.0,
        "discount": 0.0,
    }
    actual_steps = 0

    # Training loop: sample data -> calculate loss -> update params
    for _ in range(num_training_steps):
        batch = game_manager.replay_buffer.sample_batch(
            batch_size=batch_size, td_steps=td_steps, unroll_steps=unroll_steps
        )

        if batch is None:
            print("Buffer too small, skipping training step.")
            break

        params, opt_state, metrics = train_step(
            model, params, optimizer, opt_state, batch
        )

        for key in accumulated_metrics.keys():
            accumulated_metrics[key] += float(metrics[key])

        actual_steps += 1

    if actual_steps > 0:
        avg_metrics = {
            key: val / actual_steps for key, val in accumulated_metrics.items()
        }

        print(f"🎯 Average Loss this generation: {avg_metrics['total']:.4f}\n")

        wandb.log(
            {
                "Loss/Total": avg_metrics["total"],
                "Loss/Policy": avg_metrics["policy"],
                "Loss/Value": avg_metrics["value"],
                "Loss/Reward": avg_metrics["reward"],
                "Loss/Discount": avg_metrics["discount"],
            }
        )

        return params, opt_state, avg_metrics["total"]

    return params, opt_state, None
