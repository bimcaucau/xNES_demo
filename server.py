"""
Docstring for server
- Main controller
- Run the simulation loop
- Handle WebSocket communication for visualization on web
"""

import asyncio
import websockets
import json
import numpy as np
from model import PolicyNetwork
from nes import xNES
import os

# CONFIGURATION
# ====================================
GRID_SIZE = 28
EPISODE_STEPS = 50
POP_SIZE = 200
SAFE_ZONE = {'x': 14, 'y': 0, 'w': 14, 'h': 28} # the top left corner and the size from it
CHECKPOINT_FILE = "nes_checkpoint.npz"
# ====================================

# INTIALIZATION
# ====================================
policy = PolicyNetwork(input_size=2, hidden_size=[8,8], output_size=4)
optimizer = xNES(policy.total_params, POP_SIZE)
# Load the checkpoint
if os.path.exists(CHECKPOINT_FILE):
    optimizer.load(CHECKPOINT_FILE)
# ====================================

# HELPER FUNCTIONS
# ====================================
def get_fitness(final_pos): # Fitness will be +1 if inside safe zone; otherwise, -1
    x, y = final_pos
    sx, sy, sw, sh = SAFE_ZONE['x'], SAFE_ZONE['y'], SAFE_ZONE['w'], SAFE_ZONE['h']
    if sx <= x < sx + sw and sy <= y < sy + sh:
        return 1
    return -1

async def handler(websocket): # Transfer data using websocket for visualization
    print("CONNECTED")
    generation = 0
    # Default animation delay (0 means go as fast as possible)
    current_delay = 0.0
    try:
        while True:
            # 1.1 Ask for samples (genomes)
            genomes = optimizer.ask()

            # 1.2 Initialize creatures randomly
            positions = [] # list of positions
            colors = [] # list of colors
            for _ in range(POP_SIZE):
                pos = [np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)]
                positions.append(pos)
                colors.append(f"hsl({np.random.randint(0, 360)}, 70%, 50%)")

            # 2. Run the episodes step by step
            safe_center_x = int(SAFE_ZONE['x'] + SAFE_ZONE['w'] / 2)
            safe_center_y = int(SAFE_ZONE['y'] + SAFE_ZONE['h'] / 2)

            for step in range(EPISODE_STEPS):
                # Check for speed control 
                try:
                    # Check if client sent a message, wait max 0.0001 seconds
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.0001)
                    data = json.loads(msg)
                    if data.get("type") == "speed":
                        current_delay = float(data["value"])
                except asyncio.TimeoutError:
                    pass # No message sent, continue simulation
                except Exception:
                    pass # Ignore other errors to keep sim running

                # Calculate movements
                for i in range(POP_SIZE):
                    # Normalized inputs for the network
                    dx = (positions[i][0] - safe_center_x) / GRID_SIZE
                    dy = (positions[i][1] - safe_center_y) / GRID_SIZE

                    # Get action
                    action = policy.get_action(genomes[i], np.array([dx, dy]))
                    # Apply action: 0 - Up, 1 - Left, 2 - Down, 3 - Right
                    old_x, old_y = positions[i]
                    if action == 0:
                        old_y = old_y - 1
                    elif action == 1:
                        old_x = old_x - 1
                    elif action == 2:
                        old_y = old_y + 1
                    elif action == 3:
                        old_x = old_x + 1
                    # Check boundary then apply the movement
                    positions[i][0] = np.clip(old_x, 0, GRID_SIZE - 1)
                    positions[i][1] = np.clip(old_y, 0, GRID_SIZE - 1)
                
                # Send frame to client for visualization 
                # Not sending every frames and a small delay
                if current_delay > 0 or step % 2 == 0:
                    payload = {
                        "type": "update",
                        "gen": generation, 
                        "step": step,
                        "positions": np.array(positions).tolist(),
                        "colors": colors, 
                        "safe_zone": SAFE_ZONE,
                        "grid_size":GRID_SIZE
                    }
                    await websocket.send(json.dumps(payload))
                    await asyncio.sleep(max(0.001, current_delay))

            # 3. Calculate fitness at the final step - must outside the loop, due to calculating on the final step only
            fitness_scores = [get_fitness(p) for p in positions]

            # Metrics
            success_count = sum(1 for f in fitness_scores if f > 0)
            success_rate = success_count / POP_SIZE
            mean_fitness = np.mean(fitness_scores)

            # Send End of Generation stats
            end_stats = {
                "type": "stats",
                "gen": generation, 
                "success_rate": success_rate,
                "mean_fitness": float(mean_fitness)
            }
            await websocket.send(json.dumps(end_stats))
            print(f"Gen {generation}: Success Rate {success_rate*100:.1f}%")

            # 4. NES update
            optimizer.tell(fitness_scores)
            # Save periodically
            if generation > 0 and generation % 10 == 0:
                optimizer.save(CHECKPOINT_FILE)
            generation += 1
            await asyncio.sleep(0.5) # Delay between generations
    except websockets.exceptions.ConnectionClosed:
        print("DISCONNECTED")
# ====================================

async def main():
    async with websockets.serve(handler, "localhost", 4726):
        print("Server started on ws://localhost:4726")
        await asyncio.Future() # Run forever
    
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nStopping... Saving final checkpoint.")
        optimizer.save(CHECKPOINT_FILE)
        print("Goodbye!")
        
