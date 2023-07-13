import carla
import pygame
from tqdm.auto import tqdm
import random
import numpy as np


class CarEnv:

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.world = self.client.reload_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        # self.lr = 1.538900111477258 # from system identification
        self.vehicle = None
        self.actor_list = []

        # world in sync mode
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=NO_RENDERING, # False for debug
            synchronous_mode=True,
            fixed_delta_seconds=DT_))

        # waypoint init
        self.waypoint_buffer = deque(maxlen=WAYPOINT_BUFFER_LEN)

        # # same format as OpenAI gym
        # self.nb_states = 5
        # self.nb_actions = 3

        self.display = pygame.display.set_mode(
                (RES_X, RES_Y),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()

        self.file_num = 0
        # self.waypoint_chasing_index = 0

    def reset(self): # reset at the beginning of each episode "current_state = env.reset()"
        # print("call reset. ")
        tqdm.write("call reset")


        self.collision_hist = []
        self.actor_list = [] # include the vehicle and the collision sensor # no multiagent at this point

        self.spawn_point = random.choice(self.world.get_map().get_spawn_points()) # everytime set a new spawning point # ??? How about the destination?
        #self.spawn_point = self.world.get_map().get_spawn_points()[8] # fixed for testing

        new_car = False
        if self.vehicle is None:
            new_car = True

        if new_car == True:
            self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
            self.actor_list.append(self.vehicle)
        else:
            self.vehicle.set_transform(self.spawn_point)

        if new_car == True:
            transform = carla.Transform(carla.Location(x=2.5, z=0.7)) # transform for collision sensor attachment
            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            self.colsensor.listen(lambda event: self.collision_data(event)) # what's the mechanism?
            
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '{}'.format(RES_X))
            blueprint.set_attribute('image_size_y', '{}'.format(RES_Y))
            # blueprint.set_attribute('fov', '110')
            self.camera = self.world.spawn_actor(
                blueprint,
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
            self.camera.image_size_x = 1920
            self.actor_list.append(self.camera)
            self.image_queue = queue.Queue()
            self.camera.listen(self.image_queue.put)
            # self.heat_bar_image = pygame.Surface((150, 20))

            # # self.episode_start = time.time() # comment it bcs env is in sync mode
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        self.waypoint_buffer = deque(maxlen=WAYPOINT_BUFFER_LEN)
        # self.waypoint_chasing_index = 0
        self.update_waypoint_buffer(given_loc=[True, self.spawn_point.location])

        self.time = 0
        return self.get_state(), self.get_waypoint()

    def collision_data(self, event):
        self.collision_hist.append(event)

    def update_waypoint_buffer(self, given_loc = [False, None]):
        if given_loc[0]:
            car_loc = given_loc[1]
        else:
            car_loc = self.vehicle.get_location()

        self.min_distance = onp.inf
        if (len(self.waypoint_buffer) == 0):
            self.waypoint_buffer.append(self.map.get_waypoint(car_loc))

        for i in range(len(self.waypoint_buffer)):
            curr_distance = self.waypoint_buffer[i].transform.location.distance(car_loc)
            if curr_distance < self.min_distance:
                self.min_distance = curr_distance
                min_distance_index = i

        num_waypoints_to_be_added = max(0, min_distance_index - WAYPOINT_BUFFER_MID_INDEX)
        num_waypoints_to_be_added = max(num_waypoints_to_be_added, WAYPOINT_BUFFER_LEN - len(self.waypoint_buffer))

        for _ in range(num_waypoints_to_be_added):
            frontier = self.waypoint_buffer[-1]
            next_waypoints = list(frontier.next(WAYPOINT_INTERVAL))
            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, frontier)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]
            self.waypoint_buffer.append(next_waypoint)
        
        self.min_distance_index = WAYPOINT_BUFFER_MID_INDEX if min_distance_index > WAYPOINT_BUFFER_MID_INDEX else min_distance_index
        # self.waypoint_chasing_index = max(self.waypoint_chasing_index, self.min_distance_index+1)
    
    def get_state(self,):

        # collect information

        self.location = self.vehicle.get_location()
        self.location_ = onp.array([self.location.x, self.location.y, self.location.z])

        # tqdm.write("get state, loc = {}, spawn = {}".format(self.location_[:2], [self.spawn_point.location.x, self.spawn_point.location.y]))

        self.transform = self.vehicle.get_transform()
        # self.yaw = onp.array(self.transform.rotation.yaw) # float, only yaw: only along z axis # check https://d26ilriwvtzlb.cloudfront.net/8/83/BRMC_9.jpg 
        phi = self.transform.rotation.yaw*onp.pi/180 # phi is yaw

        self.velocity = self.vehicle.get_velocity()
        vx = self.velocity.x
        vy = self.velocity.y

        beta_candidate = onp.arctan2(vy, vx) - phi + onp.pi*onp.array([-2,-1,0,1,2])
        local_diff = onp.abs(beta_candidate - 0)
        min_index = onp.argmin(local_diff)
        beta = beta_candidate[min_index]

        # state = [self.velocity.x, self.velocity.y, self.yaw, self.angular_velocity.z]
        state = [
                    self.location.x, # x
                    self.location.y, # y
                    onp.sqrt(vx**2 + vy**2), # v
                    phi, # phi
                    beta, # beta
                ]

        return onp.array(state)

    def get_waypoint(self,):
        waypoints = []
        for i in range(self.min_distance_index, self.min_distance_index+FUTURE_WAYPOINTS_AS_STATE):
            waypoint_location = self.waypoint_buffer[i].transform.location
            waypoints.append([waypoint_location.x, waypoint_location.y])

        return onp.array(waypoints)

    def step(self, action): # 0:steer; 1:throttle; 2:brake; onp array shape = (3,)
        assert len(action) == 3

        if self.time >= START_TIME: # starting time
            steer_, throttle_, brake_ = action
        else:
            steer_ = 0
            throttle_ = 0.5
            brake_ = 0

        assert steer_ >= -1 and steer_ <= 1 and throttle_ <= 1 and throttle_ >= 0 and  brake_ <= 1 and brake_ >= 0

        tqdm.write(
            "steer = {0:5.2f}, throttle {1:5.2f}, brake {2:5.2f}".format(float(steer_), float(throttle_), float(brake_))
        )

        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle_), steer=float(steer_), brake=float(brake_)))

        # move a step
        for i in range(N_DT):
            self.clock.tick()
            self.world.tick() # needs to be tested! use time.sleep(??) to test
            self.time += DT_
            image_rgb = self.image_queue.get()
            draw_image(self.display, image_rgb)
            # self.display.blit(
            #     self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
            #     (8, 10))
            # self.display.blit(
            #     self.font.render('% 5d FPS (simulated)' % int(1/DT_), True, (255, 255, 255)),
            #     (8, 28))

            vel = self.vehicle.get_velocity()
            self.display.blit(
                self.font.render('Velocity = {0:.2f} m/s'.format(math.sqrt(vel.x**2 + vel.y**2)), True, (255, 255, 255)),
                (8, 10))
            
            v_offset = 25
            bar_h_offset = 75
            bar_width = 100
            for key, value in {"steering":steer_, "throttle":throttle_, "brake":brake_}.items():
                rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                pygame.draw.rect(self.display, (255, 255, 255), rect_border, 1)
                if key == "steering":
                    rect = pygame.Rect((bar_h_offset + (1+value) * (bar_width)/2, v_offset + 8), (6, 6))
                else:
                    rect = pygame.Rect((bar_h_offset + value * (bar_width), v_offset + 8), (6, 6))
                pygame.draw.rect(self.display, (255, 255, 255), rect)
                self.display.blit(
                self.font.render(key, True, (255, 255, 255)), (8, v_offset+3))

                v_offset += 18

            # heat_rect = self.heat_bar_image.get_rect()
            # self.display.blit(self.heat_bar_image, 
            #                     heat_rect, 
            #                     (0, 0, heat_rect.w/100*steer_, heat_rect.h)
            #                 )
            
            pygame.display.flip()

            if i%2 == 0 and VIDEO_RECORD and self.time >= START_TIME:
                # Save every frame
                filename = "Snaps/%05d.png" % self.file_num
                pygame.image.save(self.display, filename)
                self.file_num += 1
            

        # do we need to wait for tick (10) first?
        # no! wait_for_tick() + world.on_tick works for async mode
        # check https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#simulation-time-step
        
        self.update_waypoint_buffer()
        if DEBUG:
            past_WP = list(itertools.islice(self.waypoint_buffer, 0, self.min_distance_index))
            future_WP = list(itertools.islice(self.waypoint_buffer, self.min_distance_index+1, WAYPOINT_BUFFER_LEN-1))
            draw_waypoints(self.world, future_WP, z=0.5, color=(255,0,0))
            draw_waypoints(self.world, past_WP, z=0.5, color=(0,255,0))
            draw_waypoints(self.world, [self.waypoint_buffer[self.min_distance_index]], z=0.5, color=(0,0,255))
            # draw_waypoints(self.world, self.waypoint_buffer)

        if len(self.collision_hist) != 0:
            done = True
        else:
            done = False

        new_state = self.get_state()
        waypoints = self.get_waypoint()

        return new_state, waypoints, done, None # new_state: onp array, shape = (N,)


if __name__ == '__main__':
    pass
