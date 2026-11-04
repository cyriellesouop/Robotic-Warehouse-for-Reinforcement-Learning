import sys
import time
import gymnasium as gym
import tarware  # registers the "tarware-..." environments
from tarware.definitions import Action

RENDER = "--render" in sys.argv            # watch it in a window
LATE_PICKER = "--late-picker" in sys.argv  # send the picker only once the AGV is waiting for it

env = gym.make("tarware-tiny-3agvs-2pickers-globalobs-v1", disable_env_checker=True)
env.reset(seed=0)
w = env.unwrapped
location_of = {coords: k for k, coords in w.action_id_to_coords_map.items()}  # (y, x) -> id

agv, picker = w.agents[0], w.agents[w.num_agvs]
shelf = w.request_queue[0]
shelf_loc = location_of[(shelf.y, shelf.x)]
station_loc, station_xy = 1, w.goals[0]

# The five tasks of the order: (name, AGV target, picker needed).
tasks = [("1 travel to shelf", shelf_loc, False), ("2 load", shelf_loc, True),
         ("3 transport to station", station_loc, False), ("4 return to storage", shelf_loc, False),
         ("5 unload", shelf_loc, True)]

def waiting_for_picker():
    """The AGV reached its target and asks for load/unload: a handling task has started."""
    return agv.busy and not agv.path and agv.req_action == Action.TOGGLE_LOAD

def task_done(i):
    """End condition of task i, read from the AGV's state after a step."""
    if i in (0, 3):
        return waiting_for_picker() or (i == 0 and agv.carrying_shelf is not None) or (i == 3 and agv.carrying_shelf is None)
    if i == 1:
        return agv.carrying_shelf is not None
    if i == 2:
        return (agv.x, agv.y) == station_xy
    return agv.carrying_shelf is None

i, start, planned = 0, 0, None
for t in range(1, w.max_steps + 1):
    name, target, _ = tasks[i]
    actions = [0] * w.num_agents
    actions[agv.id - 1] = target
    # The picker goes to the shelf location for the handling tasks (late: only once the AGV waits).
    picker_wanted = tasks[i][2] or (not LATE_PICKER and i in (0, 3))
    if picker_wanted:
        actions[picker.id - 1] = shelf_loc
    env.step(actions)
    if RENDER:
        env.render()
        time.sleep(0.2)
    if planned is None and agv.path:
        planned = len(agv.path)   # planned length of a travel task = its A* path when it starts
    while i < len(tasks) and task_done(i):
        nominal = f", planned path {planned} cells" if i in (0, 2, 3) else ""
        print(f"{tasks[i][0]:<24} steps {start:>3} -> {t:>3}: {t - start:>3} steps{nominal}")
        i, start, planned = i + 1, t, None
    if i == len(tasks):
        break
if RENDER:
    input("Order finished. Press Enter to close the window...")
env.close()



#orange hexagon (6corners): AN AVG (carrier) not carrying anything
# Orange diamond (4 corners)	a picker
# Red hexagon	an AGV carrying a shelf
# Black line inside a robot	the direction the robot is facing
# Dark blue-purple squares	shelves in storage
# Teal squares	shelves currently requested (to be delivered to a station)
# Dark grey cells (right edge)	stations (delivery goals): 6 in the tiny layout
# White cells	free floor: aisles and highways

#env.step(1) give a step of observation, the rewards  ( whether it increment or decrement) and value true or false to mention whether an episode is done or not.#
#2types of environment: Action space and observation space.