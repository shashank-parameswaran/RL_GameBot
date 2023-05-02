# the above command is used to write the file PPOBot.py

# Import packages
import numpy as np
import time
import random
import sys
import pandas
import matplotlib.pyplot as plt

from functools import partial
import logging
from gym import spaces
from luxai2021.env.agent import Agent
from luxai2021.game.actions import *



class PPOBot(Agent):
    """
    This Class is a wrapper for AgentWithModel Class in agent.py.
    Some functions have been modified and some have been directly used if there is no need for changes.
    Major changes are present in the constructor, get_rewards(), get_observation() and game_start()
    """
    # Constructor called to initialize variables
    def __init__(self, mode="train", model=None):
        super().__init__()
        
        # Below variables are required as a part of the Class as they are called during training
        self.observation_space = []
        self.model = model
        self.mode = mode
        
        #self.actionSpaceUnits = []
        #self.actionSpaceCities = []
        # Below variables are used to define the action space. 
        # The action space is standardized for this game - each unit has 5 actions and each city as 3 actions
        # The idea for using a partial function has been obtained from the file match_controller.py
        self.actionSpaceUnits = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction]
        self.actionSpaceCities = [SpawnWorkerAction, SpawnCartAction, ResearchAction]
        
        # Action space is either unit's action space or the city's action space
        self.action_space = spaces.Discrete(max(len(self.actionSpaceUnits), len(self.actionSpaceCities)))
        
        
        
        # User defined variables
        self.resource_list = []
        self.unit_obs_shape = 37
        self.obs_shape = 1+self.unit_obs_shape + 4 + 2 #+ 2 + (16*16*3)
        self.last_turn_reward = 0
        
        # Observation space is common and a 47x1 matrix
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype=np.float16)

    
    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        Code for this function obtained from agent.py. No changes needed
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT
    
    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.
        """
        self.resources_last_turn = 0
        self.cities_last_turn = 0 
        self.units_last_turn = 0
        self.research_points_last_turn = 0
        self.dist_resource_last_turn = 0
        self.max_cities = 0
        self.fuel_last_turn = 0
        self.city_created_flag = False
        self.last_turn_unit_pos = []
        self.last_turn_unit_cargo = []
    
    
    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city

        """
        
        # Call and initialize the self.resource_list variable
        self.get_resources(game)
        
        # Observations related to Unit - for more details refer to get_unit_obs()
        unit_obs = self.get_unit_obs(game, unit)
        
        # Observations related to Environment - for more details refer to get_env_obs()
        env_obs = self.get_env_obs(game)
        #opp_obs = self.get_opp_obs(game) # not used - opponent observations
        
        # Observations related to Resources - for more details refer to get_resource_obs()
        resource_obs = self.get_resource_obs(game, unit) 
        
        # Not used. Idea was to pass the map information in the observations. 
        # However, it did not yield good results possibly because of a sparse matrix
        #map_info_obs = self.get_map_info(game)
        
        # Combine all observations
        all_obs = []
        all_obs.extend([1] if unit!=None else [0]) # Unit type
        all_obs.extend(unit_obs) # 37
        all_obs.extend(env_obs)  # 4
        all_obs.extend(resource_obs) # 2 
        
        
        #all_obs.extend(opp_obs) # 2
        #all_obs.extend(map_info_obs)
        #print(all_obs)
        #assert(len(all_obs)== 20)
        #[self.find_nearest_resource(unit.pos.x, unit.pos.y)] #self.get_resource_obs(game, unit)
        
        return(all_obs)
    
    def get_unit_obs(self, game, unit):
        """
        Observations related to Unit:
        Current resource value
        Distance from nearest city
        Adjacent cell resources Y/N
        Direction to nearest resource & city (OHE)
        Distance to nearest resource & city
        Adjacent cell units Y/N
        Others (can act Y/N, current tile info etc.)
        """
        if unit != None:
            unit_obs = []
            unit_obs.append(unit.type)
            dirs = [(1,0), (0,1), (-1,0), (0,-1)]
            
            # check if nearby cell has any resources
            for dir_ in dirs:
                nearby_cell = game.map.get_cell(unit.pos.x+dir_[0], unit.pos.y+dir_[1])
                if nearby_cell != None:
                    if nearby_cell.resource != None:
                        unit_obs.append(1)
                    else:
                        unit_obs.append(0)
                else:
                    unit_obs.append(0)
            
            # check if nearby cell has any units
            for dir_ in dirs:
                nearby_cell = game.map.get_cell(unit.pos.x+dir_[0], unit.pos.y+dir_[1])
                
                if nearby_cell != None:
                    if nearby_cell.has_units() and nearby_cell.city_tile == None:
                        unit_obs.append(1)
                    else:
                        unit_obs.append(0)
                else:
                    unit_obs.append(0)
                
            unit_cell = game.map.get_cell(unit.pos.x, unit.pos.y)
            unit_obs.append(1 if unit_cell.is_city_tile() else 0)
            unit_obs.append(1 if unit_cell.has_resource() else 0)
            
            unit_obs.append(unit.get_cargo_fuel_value()/100) # Some normalization wherever possible
            unit_obs.append(unit.get_cargo_space_left()/100)
            unit_obs.append(unit.get_light_upkeep()/100) #if game.is_night() else 0
            unit_obs.append(1 if unit.can_build(game.map) else 0)
            unit_obs.append(1 if unit.can_act() else 0)
            unit_obs.append(self.get_nearest_city(game, unit.pos))
            
            # add direction to nearest resource
            direction_list  = [0 for i in range(0,15)]
            direction_wood = self.direction_to_nearest_resource(game, unit, "wood")
            direction_coal = self.direction_to_nearest_resource(game, unit, "coal")
            direction_uranium = self.direction_to_nearest_resource(game, unit, "uranium")
            direction_list[direction_wood] = 1
            direction_list[direction_coal+5] = 1
            direction_list[direction_uranium+10] = 1
            unit_obs.extend(direction_list)
            
            # add direction to nearest city
            direction_list = [0,0,0,0,0]
            direction = self.direction_to_nearest_city(game, unit)
            direction_list[direction] = 1
            unit_obs.extend(direction_list)
            assert(len(unit_obs)== self.unit_obs_shape)
            return(unit_obs)
        else:
            return([0 for x in range(self.unit_obs_shape)])
    
    def get_env_obs(self, game):
        """
        Env observations:
        Night Y/N, Turns to night
        Research points
        Number of units
        -- can add in future - nearest unit from city tile
        """
        
        env_obs = []
        
        # Night time is important as bot needs to survive by consuming resources
        env_obs.append(1 if game.is_night() else 0) 
        turns_to_night = (int(game.state['turn']/40)*40 + 40 - 10 - game.state['turn'])/40 # normalized by 40
        if turns_to_night < 0:
             turns_to_night = 0
        env_obs.append(turns_to_night)
        env_obs.append(game.state['teamStates'][0]['researchPoints']/20)
        env_obs.append(len(game.state['teamStates'][0]['units'].keys())/10)
        assert(len(env_obs)==4)
        return(env_obs)
    
    def get_opp_obs(self, game):
        """
        Resources related to the opponent. Not used in this project
        """
        opp_obs = []
        opp_obs.append(game.state['teamStates'][1]['researchPoints']/20)
        opp_obs.append(len(game.state['teamStates'][1]['units'].keys())/10)
        assert(len(opp_obs)==2)
        return(opp_obs)
    
    def get_resource_obs(self, game, unit): 
        """
        Resource Observations:
        Manhattan distance to nearest resource
        -- can add in future: unit near nearest resource, amount of nearest resource
        """
        resource_obs = []
        resource_obs.append(self.find_nearest_resource(unit.pos.x, unit.pos.y) if unit!=None else 5)
        resource_obs.append(game.stats["teamStats"][self.team%2]['fuelGenerated']/100) # normalized by 100
        #resource_obs.append(sum(game.stats["teamStats"][self.team]['resourcesCollected'].values()))
        assert(len(resource_obs)==2)
        return(resource_obs)
    
    
    def get_resources(self, game):
        """
        This function obtains the resource type, cell location of all resources in the map
        """
        self.resource_list = []
        for i in range(game.map.width):
            for j in range(game.map.height):
                current_cell = game.map.get_cell(i, j)
                if current_cell.has_resource():
                    self.resource_list.append([current_cell.resource.type, i, j])
    
    def find_nearest_resource(self, x, y):
        """Find nearest resource using Manhattan distance"""
        rl = np.array(self.resource_list, dtype="object")
        rl = rl[:,1:]
        manhat = np.abs(np.array([x, y]) - np.array(rl[1:], dtype="int64"))
        nearest_resource_dist = np.min(np.sum(manhat, axis=1))
        return(nearest_resource_dist)
        

    def get_actions(self, game):
        return(0)
        #return([self.actionSpaceUnits, self.actionSpaceCities])
    
    
    def get_map_info(self, game):
        """
        Not used. Idea was to pass the map information in the observations. 
        However, it did not yield good results possibly because of a sparse matrix
        """
        map_info_obs = []
        #print(game.map.height, game.map.width)
        for i in range(game.map.height):
            for j in range(game.map.width):
                cell = game.map.get_cell(i, j)
                if cell.has_resource():
                    map_info_obs.extend([1,0,0])
                elif cell.is_city_tile():
                    map_info_obs.extend([0,1,0])
                elif cell.has_units():
                    map_info_obs.extend([0,0,1])
                else:
                    map_info_obs.extend([0,0,0])
        #print(game.map.height*game.map.width)
        assert len(map_info_obs)==16*16*3#game.map.height*game.map.width
        return(map_info_obs)
    
    def get_nearest_city(self, game, pos):
        """
        Find the nearest city from the unit
        """
        dist_list = [10]
        cities = list(game.cities.values())
        try:
            if len(cities) > 0:
                if len(cities[self.team%2].city_cells) > 0: # self.team%2 gets the team id
                    for city in list(game.cities.values())[self.team%2].city_cells:
                        manhat_dist = np.abs(city.pos.x - pos.x) + np.abs(city.pos.y - pos.y)
                        dist_list.append(manhat_dist)
            else:
                return(10)
        except:
            pass
        return(min(dist_list))

    def direction_to_nearest_city(self, game, unit):
        """
        Direction to nearest city - outputs 0,1,2,3,4 based on direction
        """
        
        # Idea for using the below dictionary obtained from https://www.kaggle.com/code/glmcdona/reinforcement-learning-openai-ppo-with-python-game
        mapping = {
                Constants.DIRECTIONS.CENTER: 0,
                Constants.DIRECTIONS.NORTH: 1,
                Constants.DIRECTIONS.WEST: 2,
                Constants.DIRECTIONS.SOUTH: 3,
                Constants.DIRECTIONS.EAST: 4,
            }
        dist = []
        cities = game.cities.values()
        for city in cities:
            if city.team == unit.team:
                #print("num cities:", len(city.city_cells))
                for citycell in city.city_cells:
                    #print("City Pos:", citycell.pos.x, citycell.pos.y)
                    x_dif = np.abs(unit.pos.x - citycell.pos.x)
                    y_dif = np.abs(unit.pos.y - citycell.pos.y)
                    manhat = x_dif+y_dif
                    dist.append(manhat)
                min_dist_ind = np.argmin(np.array(manhat))
                direction = unit.pos.direction_to(city.city_cells[min_dist_ind].pos)
                #print("unit pos:", unit.pos.x, unit.pos.y)
                #print("direction: ", direction)
                return(mapping[direction])
        return(0)
    
    def direction_to_nearest_resource(self, game, unit, typ):
        """
        Direction to nearest resource - outputs 0,1,2,3,4 based on direction
        """
        mapping = {
                Constants.DIRECTIONS.CENTER: 0,
                Constants.DIRECTIONS.NORTH: 1,
                Constants.DIRECTIONS.WEST: 2,
                Constants.DIRECTIONS.SOUTH: 3,
                Constants.DIRECTIONS.EAST: 4,
            }
        rl = np.array(self.resource_list, dtype="object")
        
        #print("Before:", len(rl))
        #print(rl)
        rl = rl[np.in1d(rl[:, 0], np.asarray([typ]))]
        #print("After:", len(rl))
        #print(rl)
        rl = rl[:,1:]
        if len(rl)==0:
            return(0)
        manhat = np.abs(np.array([unit.pos.x, unit.pos.y]) - np.array(rl, dtype="int64"))
        nearest_resource_ind = np.argmin(np.sum(manhat, axis=1))
        nearest_resource = self.resource_list[nearest_resource_ind][1:]
        cell = game.map.get_cell(nearest_resource[0], nearest_resource[1])
        direction = unit.pos.direction_to(cell.pos)
        return(mapping[direction])    
    
    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        
        All rewards are calcualted for just one team at a time
        Below rewards are calculated:
        Inc. Cities from last turn
        Inc. Units from last turn
        Inc. research points from last turn
        Inc. resources from last turn
        Unit roaming penalty (with full cargo)
        Unit distance penalty from city at night
        Penalty for not building city when cargo is full
        Penalty for Inc. distance from resource from last turn
        Penalty for building city at night
        
        Some rewards were tested but unused and retained to show progress
        """
        
        # Units in current turn
        current_turn_unit_list = list(game.state["teamStates"][0]["units"].values())
        
        if is_game_error == True:
            return(-50)
        
        if is_new_turn == False and is_game_finished==False:
            return(0)
        
        # Number of player units
        player_units = len(game.state['teamStates'][self.team%2]['units'].keys())
        
        # Number of player cities
        player_cities = 0
        for city in game.cities.values():
            if city.team == self.team:
                player_cities +=1
        #player_cities = len(game.cities.keys())
        
        # Player research points
        player_research_points = game.state['teamStates'][self.team%2]['researchPoints']
        
        # Sum of distances from resources for all units
        player_dist_resource = 0
        for unit in current_turn_unit_list:
            dist = self.find_nearest_resource(unit.pos.x, unit.pos.y)
            if dist<=1:
                dist = 0
            if game.is_night()==True:
                player_dist_resource += (1-dist)*20
            else:
                player_dist_resource += (1-dist)*5
        
        #print("Turn:", game.state["turn"], "Res Dist:", inc_dist_resource, "Pen:", inc_dist_resource)
        
        # sum (incremental resources from last turn)
        player_total_resource = 0
        for i in range(len(current_turn_unit_list)):
            try:
                player_resource_current = sum(current_turn_unit_list[i].cargo.values())
                diff = (player_resource_current-self.last_turn_unit_cargo[i]) # last_turn_unit_cargo defined at end and initialized at game start
                if diff == 100: # if diff is 100 then it has created a city. No need to penalize this
                    diff = 0
                player_total_resource += diff
            except:
                pass
        
        # Unit roaming penalty and penalty for not building a city with full cargo
        player_roaming_penalty = 0
        not_build_city_penalty = 0
        if len(self.last_turn_unit_pos) == len(current_turn_unit_list):
            #print("same units")
            for i in range(len(current_turn_unit_list)):
                if ((sum(current_turn_unit_list[i].cargo.values()) == 100) & (self.last_turn_unit_cargo[i] == 100)):
                    if self.get_nearest_city(game, current_turn_unit_list[i].pos) > self.get_nearest_city(game, self.last_turn_unit_pos[i]):
                        #print("farther")
                        player_roaming_penalty += 20
                    unit_cell = game.map.get_cell(self.last_turn_unit_pos[i].x, self.last_turn_unit_pos[i].y)
                    if (unit_cell.resource == None) & (unit_cell.city_tile==None):
                        not_build_city_penalty +=40
        
        # Night time penalties - Units need to consume available resources at night or it will get eliminated
        # distance from city and cargo space left (less cargo - more likely to get eliminated at night)
        turns_to_night = int(game.state['turn']/40)*40 + 40 - 10 - game.state['turn']
        player_dist_night = 0
        player_night_space = 0
        for unit in current_turn_unit_list:
            player_dist_night += self.get_nearest_city(game, unit.pos)
            player_night_space += 0.1*unit.get_cargo_space_left()
        
        # Increase penalties as night time approaches
        if (turns_to_night >=5) and (turns_to_night<10):
            player_dist_night *=2
            player_night_space *=2
        elif (turns_to_night >0) and (turns_to_night<5):
            player_dist_night *=4
            player_night_space *=4
        elif turns_to_night==0:
            player_dist_night *= 7
            player_night_space *=7
        else:
            player_dist_night=0
            player_night_space =0
        
        #print("Turn:", game.state["turn"], "Night Dist:",player_dist_night, "Units:", player_units)
        
        # Penalty for building a city at night time
        # If a unit builds a city at night time, it loses those resources and it is left with no resources to survive the night
        build_city_penalty = 0
        if self.cities_last_turn < player_cities:
            if turns_to_night<=3:
                not_build_city_penalty=0
                build_city_penalty +=50
        
        # Resources deposited in the city
        player_fuel = game.stats["teamStats"][self.team%2]['fuelGenerated']
        #min_fuel = (player_fuel- 330) * 0.5
        #print(min_fuel)
            
        
        #if game.is_night()==True:
            #player_total_resource = player_total_resource *1.5
        
        # Get Incremental rewards in turn
        inc_cities = player_cities - self.cities_last_turn
        inc_units = player_units - self.units_last_turn
        inc_research_points = player_research_points - self.research_points_last_turn
        inc_dist_resource = player_dist_resource - self.dist_resource_last_turn
        inc_fuel = player_fuel - self.fuel_last_turn
        inc_resources = player_total_resource - self.resources_last_turn
        inc_resources = 0 if inc_resources < 0 else inc_resources # when building a city or consume resources during night
        
        #inc_fuel = 0 if inc_resources < 0 else inc_resources
        #inc_cities = -0.1 if inc_cities < 0 else inc_cities
        #inc_units = -0.1 if inc_units < 0 else inc_units
        #inc_research_points = -1 if inc_research_points < 0 else inc_research_points
        #print("player fuel:", player_fuel)
        #print("Last turn:", self.fuel_last_turn)
        #print("Inc:", inc_fuel*2)
        
        # Total reward for current turn
        player_current_reward = sum([
                                     # Positive rewards
                                     inc_cities*300,
                                     inc_units*100,
                                     inc_research_points*10,
                                     inc_resources,
                                     inc_fuel*2, # subs with inc fuel
                                    
                                     # Negative Rewards
                                     -player_roaming_penalty,
                                     -player_dist_night,
                                     -player_night_space,
                                     -build_city_penalty,
                                     -not_build_city_penalty,
                                     inc_dist_resource # already a negative value - refer player_dist_resource above
                                    ])

        
        # Assign values for previous turn metrics to use for incremental calculations for next step
        self.cities_last_turn = player_cities
        self.units_last_turn = player_units
        self.research_points_last_turn = player_research_points
        self.dist_resource_last_turn = player_dist_resource
        self.fuel_last_turn = player_fuel
        self.resources_last_turn = player_total_resource
        self.max_cities = max(self.max_cities, player_cities)
        
        self.last_turn_unit_cargo = []
        self.last_turn_unit_pos = []
        for unit in current_turn_unit_list:
            self.last_turn_unit_pos.append(unit.pos)
            self.last_turn_unit_cargo.append(sum(list(unit.cargo.values())))
        
        # Logs - used only for testing the codes and printing values
        cities_a = sum([1 for city in game.cities.values() if city.team%2==0])
        cities_b = sum([1 for city in game.cities.values() if city.team%1==0])
        units_a = sum([1 for unit in game.state["teamStates"][self.team%2]["units"].values()])
        units_b = sum([1 for unit in game.state["teamStates"][self.team%1]["units"].values()])
        
        
        
        """
        if self.team==0:
            print("End:", round(player_current_reward, 1),
                     "Turn:", game.state["turn"], 
                     "Cities:", cities_a, cities_b,
                      "Units:", units_a, units_b,
                      "MaxCity:", self.max_cities,
                      "Res:", player_research_points,
                      "city:", inc_cities*200, 
                      "Unit:", inc_units*100, 
                      "RewResh:", inc_research_points*10, 
                      "RR:", round(inc_resources,1), 
                      "Fuel:", inc_fuel, 
                      "Roam:", -player_roaming_penalty, 
                      "CityPen:", -(build_city_penalty+not_build_city_penalty),
                      "Dist:", -inc_dist_resource,
                     )
        """
        if is_game_finished and random.random() <= 0.05 and self.team==0:
            print("End:", round(player_current_reward, 1),
                 "Turn:", game.state["turn"], 
                  "Seed:", game.configs["seed"],
                 "Cities:", cities_a, cities_b,
                  "Units:", units_a, units_b,
                  "MaxCity:", self.max_cities,
                  "Res:", player_research_points,
                  "city:", inc_cities*300, 
                  "Unit:", inc_units*100, 
                  #"RewResh:", inc_research_points*10, 
                  #"RewReso:", round(inc_resources,1), 
                  "Fuel:", inc_fuel, 
                  #"Roam:", -player_roaming_penalty, 
                  "CityPen:", -(build_city_penalty+not_build_city_penalty),
                  "Dist:", -inc_dist_resource,
                  "NSP:", -player_night_space
                 )
        return(player_current_reward)
    
    
    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference.
        Returns: Array of actions to perform.
        Obtained from agent.py directly. No changes needed
        """
        start_time = time.time()
        actions = []
        new_turn = True

        # Inference the model per-unit
        units = game.state["teamStates"][team]["units"].values()
        for unit in units:
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, new_turn)
                action_code, _states = self.model.predict(obs, deterministic=False)
                if action_code is not None:
                    actions.append(
                        self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team))
                new_turn = False

        # Inference the model per-city
        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    if city_tile.can_act():
                        obs = self.get_observation(game, None, city_tile, city.team, new_turn)
                        action_code, _states = self.model.predict(obs, deterministic=False)
                        if action_code is not None:
                            actions.append(
                                self.action_code_to_action(action_code, game=game, unit=None, city_tile=city_tile,
                                                           team=city.team))
                        new_turn = False

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
                  file=sys.stderr)
        
        return actions
    
    
    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        Returns: An action.
        
        Obtained from https://www.kaggle.com/code/glmcdona/reinforcement-learning-openai-ppo-with-python-game as this is standardized and does not affect the model output. 
        """
        # Map actionCode index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y
            
            if city_tile != None:
                action =  self.actionSpaceCities[action_code%len(self.actionSpaceCities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )

                # If the city action is invalid, default to research action automatically
                if not action.is_valid(game, actions_validated=[]):
                    action = ResearchAction(
                        game=game,
                        unit_id=unit.id if unit else None,
                        unit=unit,
                        city_id=city_tile.city_id if city_tile else None,
                        citytile=city_tile,
                        team=team,
                        x=x,
                        y=y
                    )
            else:
                action =  self.actionSpaceUnits[action_code%len(self.actionSpaceUnits)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            
            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None
    
    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
        actionCode: Index of action to take into the action array.
        Obtained from https://www.kaggle.com/code/glmcdona/reinforcement-learning-openai-ppo-with-python-game as this is standardized and does not affect the model output. 
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)
