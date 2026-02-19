import jax
import optax
from functools import partial
from loss_function import loss_function


@partial(jax.jit, static_argnums=(0, 2))
def train_step(model, params, optimizer, opt_state, batch):
    # Calculate derivatives
    grad_fn = jax.value_and_grad(loss_function)
    loss_value, grads = grad_fn(params, model, batch)

    # Update params
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss_value


def training_loop(
    model,
    params,
    optimizer,
    opt_state,
    num_gens,
    games_per_gen,
    num_training_steps,
    batch_size,
    unroll_steps,
    game_manager,
):
    print("\n========== Starting Training ==========")
    print(f"Training for {num_gens} generations...\n")

    loss_history = []

    for gen in range(num_gens):
        print(f"===== Generation {gen + 1} =====")

        print(f"Playing {games_per_gen} games...")
        for _ in range(games_per_gen):
            game_manager.params = params
            game_manager.play_single_episode(max_episode_length=100)

        print(f"Training for {num_training_steps} steps...")
        total_gen_loss = 0.0

        # Training loop:
        #   sample data -> calculate loss -> update params
        for _ in range(num_training_steps):
            batch = game_manager.replay_buffer.sample_batch(batch_size, unroll_steps)

            if batch is None:
                print("Buffer too small, skipping training step.")
                break

            params, opt_state, loss = train_step(
                model, params, optimizer, opt_state, batch
            )
            total_gen_loss += float(loss)

        if batch is not None:
            avg_loss = total_gen_loss / num_training_steps
            loss_history.append(avg_loss)
            print(f"Average Loss this generation: {avg_loss:.4f}")

    return loss_history
