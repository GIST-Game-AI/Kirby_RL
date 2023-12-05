import datetime
from pathlib import Path
from pyboy.pyboy import *
from gym.wrappers import FrameStack, NormalizeObservation
from AISettings.AISettingsInterface import AISettingsInterface
from AISettings.MarioAISettings import MarioAI
from AISettings.KirbyAISettings import KirbyAI
from MetricLogger import MetricLogger
from agent import AIPlayer
from wrappers import SkipFrame, ResizeObservation
import sys
from CustomPyBoyGym import CustomPyBoyGym
from functions import alphanum_key
from dotenv import load_dotenv
from utils import get_response_from_gpt

load_dotenv()
"""
  Variables
"""
episodes = 10000
# gym variables  documentation: https://docs.pyboy.dk/openai_gym.html#pyboy.openai_gym.PyBoyGymEnv
observation_types = ["raw", "tiles", "compressed", "minimal"]
observation_type = observation_types[1]
action_types = ["press", "toggle", "all"]
action_type = action_types[0]
gameDimentions = (20, 16)
frameStack = 4
quiet = False
train = False
playtest = False

gpt_bool = False
if len(sys.argv) >= 2:
    gpt_bool = sys.argv[1]
gpt_cooltime = 300
gpt_tick = 20
gpt_prompt = """
Game: {game_name}
Current State: {{ Kirby's Health: {kirby_health}, Boss Health: {boss_health}, Score: {score}, Key-down: {key_down}}}
"""

"""
  Choose game
"""
gamesFolder = Path("games")
games = [os.path.join(gamesFolder, f) for f in os.listdir(gamesFolder) if
         (os.path.isfile(os.path.join(gamesFolder, f)) and f.endswith(".gb"))]
gameNames = [f.replace(".gb", "") for f in os.listdir(gamesFolder) if
             (os.path.isfile(os.path.join(gamesFolder, f)) and f.endswith(".gb"))]

print("Avaliable games: ", games)
for cnt, gameName in enumerate(games, 1):
    sys.stdout.write("[%d] %s\n\r" % (cnt, gameName))

choice = int(input("Select game[1-%s]: " % cnt)) - 1
game = games[choice]
gameName = gameNames[choice]

"""
  Choose mode
"""
modes = ["Evaluate (HEADLESS)", "Evaluate (UI)",
         "Train (HEADLESS)", "Train (UI)", "Playtest (UI)"]
for cnt, modeName in enumerate(modes, 1):
    sys.stdout.write("[%d] %s\n\r" % (cnt, modeName))

mode = int(input("Select mode[1-%s]: " % cnt)) - 1

if mode == 0:
    quiet = True
    train = False
elif mode == 1:
    quiet = False
    train = False
elif mode == 2:
    quiet = True
    train = True
elif mode == 3:
    quiet = False
    train = True
elif mode == 4:
    quiet = False
    playtest = True

"""
  Logger
"""
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir = Path("checkpoints") / gameName / now
save_dir2 = Path("checkpoints") / gameName / (now + "-2")
save_dir_eval = Path("checkpoints") / gameName / (now + "-eval")
save_dir_boss = Path("checkpoints") / gameName / (now + "-boss")
checkpoint_dir = Path("checkpoints") / gameName

"""
  Load emulator
"""
pyboy = PyBoy(game, window_type="headless" if quiet else "SDL2", window_scale=3, debug=False, game_wrapper=True)

"""
  Load enviroment
"""
aiSettings = AISettingsInterface()
if pyboy.game_wrapper().cartridge_title == "SUPER MARIOLAN":
    aiSettings = MarioAI()
if pyboy.game_wrapper().cartridge_title == "KIRBY DREAM LA":
    aiSettings = KirbyAI()

env = CustomPyBoyGym(pyboy, observation_type=observation_type)
env.setAISettings(aiSettings)  # use this settings
filteredActions = aiSettings.GetActions()  # get possible actions
kirby_boss_filteredActions = aiSettings.GetActions()
print("Possible actions: ", [[WindowEvent(i).__str__() for i in x] for x in filteredActions])

"""
  Apply wrappers to enviroment
"""
env = SkipFrame(env, skip=4)
env = ResizeObservation(env, gameDimentions)  # transform MultiDiscreate to Box for framestack
env = NormalizeObservation(env)  # normalize the values
env = FrameStack(env, num_stack=frameStack)

"""
  Load AI players
"""
bossAiPlayer = AIPlayer((frameStack,) + gameDimentions, len(filteredActions), save_dir_boss, now,
                        aiSettings.GetBossHyperParameters())

# filter 2 actions for kirby(left+right, down+else)
if pyboy.game_wrapper().cartridge_title == "KIRBY DREAM LA":
    entry_list = list()
    for action_type in filteredActions:
        if len(action_type) != 1:
            for entry in action_type:
                if WindowEvent(entry).__str__() == "PRESS_ARROW_DOWN":
                    entry_list.append(action_type)
                    continue

            if WindowEvent(action_type[0]).__str__() == "PRESS_ARROW_LEFT" and WindowEvent(
                    action_type[1]).__str__() == "PRESS_ARROW_RIGHT":
                entry_list.append(action_type)

    for entry in entry_list:
        filteredActions.remove(entry)

print("Kirby platform Possible actions: ", [[WindowEvent(i).__str__() for i in x] for x in filteredActions])
aiPlayer = AIPlayer((frameStack,) + gameDimentions, len(filteredActions), save_dir, now,
                    aiSettings.GetHyperParameters())

print("Kirby platform 2 Possible actions: ", [[WindowEvent(i).__str__() for i in x] for x in filteredActions])
aiPlayer2 = AIPlayer((frameStack,) + gameDimentions, len(filteredActions), save_dir2, now,
                     aiSettings.GetHyperParameters())

if mode < 2:  # evaluate
    # load model
    folderList = [name for name in os.listdir(checkpoint_dir) if
                  os.path.isdir(checkpoint_dir / name) and len(os.listdir(checkpoint_dir / name)) != 0]

    if len(folderList) == 0:
        print("No models to load in path: ", save_dir)
        quit()

    for cnt, fileName in enumerate(folderList, 1):
        sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))

    choice = int(input("Select folder with platformer model[1-%s]: " % cnt)) - 1
    folder = folderList[choice]
    print(folder)

    fileList = [f for f in os.listdir(checkpoint_dir / folder) if f.endswith(".chkpt")]
    fileList.sort(key=alphanum_key)
    if len(fileList) == 0:
        print("No models to load in path: ", folder)
        quit()

    modelPath = checkpoint_dir / folder / fileList[-1]
    aiPlayer.loadModel(modelPath)

    choice = int(input("Select folder with platformer model 2[1-%s]: " % cnt)) - 1
    folder = folderList[choice]
    print(folder)

    fileList = [f for f in os.listdir(checkpoint_dir / folder) if f.endswith(".chkpt")]
    fileList.sort(key=alphanum_key)
    if len(fileList) == 0:
        print("No models to load in path: ", folder)
        quit()

    modelPath2 = checkpoint_dir / folder / fileList[-1]
    aiPlayer2.loadModel(modelPath2)

    choice = int(
        input("Select folder with boss model[1-%s] (if not using boss model select same as previous): " % cnt)) - 1
    folder = folderList[choice]
    print(folder)

    fileList = [f for f in os.listdir(checkpoint_dir / folder) if f.endswith(".chkpt")]
    fileList.sort(key=alphanum_key)
    if len(fileList) == 0:
        print("No models to load in path: ", folder)
        quit()

    bossModelPath = checkpoint_dir / folder / fileList[-1]
    bossAiPlayer.loadModel(bossModelPath)

"""
  Main loop
"""

if train:
    if len(sys.argv) >= 3:
        fileList = [f for f in os.listdir(Path("checkpoints") / gameName / sys.argv[2]) if f.endswith(".chkpt")]
        fileList.sort(key=alphanum_key)
        if len(fileList) == 0:
            print("No models to load in path: ", sys.argv[2])
            quit()
        modelPath = Path("checkpoints") / gameName / sys.argv[2] / fileList[-1]
        print("load model for aiPlayer: ", modelPath)
        aiPlayer.loadModel(modelPath)

        if len(sys.argv) >= 4:
            fileList = [f for f in os.listdir(Path("checkpoints") / gameName / sys.argv[3]) if f.endswith(".chkpt")]
            fileList.sort(key=alphanum_key)
            if len(fileList) == 0:
                print("No models to load in path: ", sys.argv[3])
                quit()
            modelPath = Path("checkpoints") / gameName / sys.argv[3] / fileList[-1]
            print("load model for bossAiPlayer: ", modelPath)
            bossAiPlayer.loadModel(modelPath)

            if len(sys.argv) >= 5:
                fileList = [f for f in os.listdir(Path("checkpoints") / gameName / sys.argv[4]) if f.endswith(".chkpt")]
                fileList.sort(key=alphanum_key)
                if len(fileList) == 0:
                    print("No models to load in path: ", sys.argv[4])
                    quit()
                modelPath = Path("checkpoints") / gameName / sys.argv[4] / fileList[-1]
                print("load model for aiPlayer: ", modelPath)
                aiPlayer2.loadModel(modelPath)
    pyboy.set_emulation_speed(0)
    save_dir.mkdir(parents=True)
    save_dir_boss.mkdir(parents=True)
    save_dir2.mkdir(parents=True)
    logger = MetricLogger(save_dir_boss)
    aiPlayer.saveHyperParameters()
    aiPlayer2.saveHyperParameters()
    bossAiPlayer.saveHyperParameters()

    print("Training mode")
    print("Total Episodes: ", episodes)
    aiPlayer.net.train()
    aiPlayer2.net.train()
    bossAiPlayer.net.train()

    player = aiPlayer
    for e in range(episodes):
        observation = env.reset()
        start = time.time()
        actionBool = True
        firstEpisodeBool = True
        while True:

            if firstEpisodeBool and aiSettings.IsBossActive(pyboy):
                firstEpisodeBool = False

            player_type = "platform"
            if aiSettings.IsBossActive(pyboy):
                player = bossAiPlayer
                player_type = "boss"
            else:
                player = aiPlayer
                if (not firstEpisodeBool) and pyboy.game_wrapper().cartridge_title == "KIRBY DREAM LA":
                    player = aiPlayer2
            # Make action based on current state
            actionId = player.act(observation)
            if pyboy.game_wrapper().cartridge_title == "KIRBY DREAM LA" and aiSettings.IsBossActive(
                    pyboy):  # for kirby boss
                if actionBool:
                    actions = kirby_boss_filteredActions[0]
                    actionBool = False
                else:
                    actions = kirby_boss_filteredActions[1]
                    actionBool = True
            else:
                actions = filteredActions[actionId]
            # Agent performs action and moves 1 frame
            next_observation, reward, done, info = env.step(actions)

            # Remember
            player.cache(observation, next_observation, actionId, reward, done)
            # Learn
            q, loss = player.learn(player_type)
            # Logging
            logger.log_step(reward, loss, q, player.scheduler.get_last_lr())
            # Update state
            observation = next_observation

            if done or time.time() - start > 500:
                break

        logger.log_episode()
        logger.record(episode=e, epsilon=player.exploration_rate, stepsThisEpisode=player.curr_step,
                      maxLength=aiSettings.GetLength(pyboy))

    aiPlayer.save("platform")
    bossAiPlayer.save("boss")
    env.close()
elif not train and not playtest:
    print("Evaluation mode")
    pyboy.set_emulation_speed(1)

    save_dir_eval.mkdir(parents=True)
    logger = MetricLogger(save_dir_eval)

    aiPlayer.exploration_rate = 0
    aiPlayer.net.eval()

    aiPlayer2.exploration_rate = 0
    aiPlayer2.net.eval()

    bossAiPlayer.exploration_rate = 0
    bossAiPlayer.net.eval()

    player = aiPlayer
    for e in range(episodes):
        observation = env.reset()
        actionBool = True
        firstEpisodeBool = True
        cur_tick = 0
        last_tick = 0
        gpt_remain_tick = 0
        gpt_answer = None
        prev_input = None
        gpt_action=None
        while True:
            if firstEpisodeBool and aiSettings.IsBossActive(pyboy):
                firstEpisodeBool = False

            if aiSettings.IsBossActive(pyboy):
                player = bossAiPlayer
            else:
                player = aiPlayer
                if not firstEpisodeBool and pyboy.game_wrapper().cartridge_title == "KIRBY DREAM LA":
                    player = aiPlayer2

            cur_tick += 1
            if gpt_remain_tick != 0:
                print("current remain tick: {}".format(gpt_remain_tick))
                gpt_remain_tick -= 1
            else:
                gpt_action = None

            if cur_tick - last_tick >= gpt_cooltime and gpt_bool:
                pyboy.botsupport_manager().screen().screen_image().save(
                    'gpt_screenshot.png')
                pyboy._pause()
                try:
                    print("gpt activate for {} tick".format(cur_tick))
                    real_prompt = gpt_prompt.format(game_name='Kirby\'s Dream Land',
                                                    kirby_health=str(aiSettings.GetGameState(pyboy).health),
                                                    boss_health=str(aiSettings.GetGameState(pyboy).boss_health),
                                                    score=str(aiSettings.GetGameState(pyboy).score),
                                                    key_down=str([WindowEvent(i).__str__() for i in prev_input]))
                    print("gpt prompt is {}".format(real_prompt))
                    gpt_result = get_response_from_gpt('./gpt_screenshot.png', real_prompt)
                    print("gpt result is {}".format(gpt_result))
                    gpt_result = gpt_result[list(gpt_result.keys())[-1]]
                    print("gpt action will be {}".format(gpt_result))
                    if len(gpt_result) == 1:
                        gpt_action = list()
                        if gpt_result[0] == 'PRESS_BUTTON_A':
                            gpt_action.append(5)
                        elif gpt_result[0] == 'PRESS_BUTTON_B':
                            gpt_action.append(6)
                        elif gpt_result[0] == 'PRESS_ARROW_UP':
                            gpt_action.append(1)
                        elif gpt_result[0] == 'PRESS_ARROW_DOWN':
                            gpt_action.append(2)
                        elif gpt_result[0] == 'PRESS_ARROW_LEFT':
                            gpt_action.append(4)
                        elif gpt_result[0] == 'PRESS_ARROW_RIGHT':
                            gpt_action.append(3)
                    else:
                        tmp_first = None
                        tmp_second = None
                        if gpt_result[0] == 'PRESS_BUTTON_A':
                            tmp_first = 5
                        elif gpt_result[0] == 'PRESS_BUTTON_B':
                            tmp_first = 6
                        elif gpt_result[0] == 'PRESS_ARROW_UP':
                            tmp_first = 1
                        elif gpt_result[0] == 'PRESS_ARROW_DOWN':
                            tmp_first = 2
                        elif gpt_result[0] == 'PRESS_ARROW_LEFT':
                            tmp_first = 4
                        elif gpt_result[0] == 'PRESS_ARROW_RIGHT':
                            tmp_first = 3
                        if gpt_result[0] == 'PRESS_BUTTON_A':
                            tmp_second = 5
                        elif gpt_result[0] == 'PRESS_BUTTON_B':
                            tmp_second = 6
                        elif gpt_result[0] == 'PRESS_ARROW_UP':
                            tmp_second = 1
                        elif gpt_result[0] == 'PRESS_ARROW_DOWN':
                            tmp_second = 2
                        elif gpt_result[0] == 'PRESS_ARROW_LEFT':
                            tmp_second = 4
                        elif gpt_result[0] == 'PRESS_ARROW_RIGHT':
                            tmp_second = 3
                        gpt_action = (tmp_first, tmp_second)
                    last_tick = cur_tick
                    gpt_remain_tick = gpt_tick
                except:
                    print("error from gpt4V. Agent will use action from RL model.")
                pyboy._unpause()

            actionId = player.act(observation)
            if pyboy.game_wrapper().cartridge_title == "KIRBY DREAM LA" and aiSettings.IsBossActive(
                    pyboy):  # for kirby boss
                if actionBool:
                    action = kirby_boss_filteredActions[0]
                    actionBool = False
                else:
                    action = kirby_boss_filteredActions[1]
                    actionBool = True
            else:
                action = filteredActions[actionId]
            if gpt_remain_tick != 0:
                print("RL's action: {}".format(action))
                print("GPT's action: {}".format(gpt_action))
                action = gpt_action
            prev_input = action
            next_observation, reward, done, info = env.step(action)

            logger.log_step(reward, 1, 1, 1)

            # print("Reward: ", reward)
            # print("Action: ", [WindowEvent(i).__str__() for i in action])
            aiSettings.PrintGameState(pyboy)

            observation = next_observation

            # print(reward)
            if done:
                break

        logger.log_episode()
        logger.record(episode=e, epsilon=player.exploration_rate, stepsThisEpisode=player.curr_step,
                      maxLength=aiSettings.GetLength(pyboy))
    env.close()

elif playtest:
    pyboy.set_emulation_speed(1)
    env.reset()
    print("Playtest mode")
    tmp = 0
    while True:
        tmp += 1
        print("tick: ", tmp)
        previousGameState = aiSettings.GetGameState(pyboy)
        env.pyboy.tick()

        print("Reward: ", aiSettings.GetReward(previousGameState, pyboy))
        print("Real max length: ", aiSettings.GetLength(pyboy))
        aiSettings.PrintGameState(pyboy)

        if env.game_wrapper.game_over():
            break
