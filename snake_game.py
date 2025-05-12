import pygame
import random
import numpy as np

class SnakeGame:
    def __init__(self, width=450, height=450, block_size=10, difficulty=25):
        pygame.init()
        self.width = width
        self.height = height
        self.block_size = block_size
        self.difficulty = difficulty
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Eater RL')
        self.clock = pygame.time.Clock()

        # Colors
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)

        # For circle movement detection
        self.recent_positions = []
        self.max_recent = 20

        self.reset()

    def reset(self):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = [random.randrange(1, self.width // self.block_size) * self.block_size,
                         random.randrange(1, self.height // self.block_size) * self.block_size]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.start_pos = self.snake_pos
        self.consecutive_food = 0
        self.recent_positions = []
        return self.get_state_as_array()

    def is_dead_end(self):
        next_pos = list(self.snake_pos)
        if self.direction == 'UP':
            next_pos[1] -= self.block_size
        elif self.direction == 'DOWN':
            next_pos[1] += self.block_size
        elif self.direction == 'LEFT':
            next_pos[0] -= self.block_size
        elif self.direction == 'RIGHT':
            next_pos[0] += self.block_size
        
        if (next_pos[0] < 0 or next_pos[0] >= self.width or
            next_pos[1] < 0 or next_pos[1] >= self.height or
            next_pos in self.snake_body[1:]):
            return True
        return False

    def step(self, action):
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        new_direction = directions[action]

        if (new_direction == 'UP' and self.direction != 'DOWN') or \
           (new_direction == 'DOWN' and self.direction != 'UP') or \
           (new_direction == 'LEFT' and self.direction != 'RIGHT') or \
           (new_direction == 'RIGHT' and self.direction != 'LEFT'):
            self.direction = new_direction

        if self.direction == 'UP':
            self.snake_pos[1] -= self.block_size
        elif self.direction == 'DOWN':
            self.snake_pos[1] += self.block_size
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= self.block_size
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += self.block_size

        self.snake_body.insert(0, list(self.snake_pos))

        reward = -0.01
        done = False

        if self.snake_pos == self.food_pos:
            self.score += 1
            reward = 100
            self.food_spawn = False
            self.consecutive_food += 1
        else:
            self.snake_body.pop()

        if not self.food_spawn:
            self.food_pos = [random.randrange(1, self.width // self.block_size) * self.block_size,
                             random.randrange(1, self.height // self.block_size) * self.block_size]
        self.food_spawn = True

        reward += 10 * self.consecutive_food

        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.width or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= self.height or
            self.snake_pos in self.snake_body[1:]):
            done = True
            reward = -20
            return self.get_state_as_array(), reward, done

        food_distance = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
        reward += max(0, 10 - food_distance // self.block_size)

        if self.is_dead_end():
            reward -= 10

        if self.snake_pos == self.snake_body[1]:
            reward -= 1

        if self.snake_pos not in self.snake_body[1:]:
            reward += 1
        else:
            reward -= 5

        distance_travelled = abs(self.snake_pos[0] - self.start_pos[0]) + abs(self.snake_pos[1] - self.start_pos[1])
        reward += distance_travelled // 10

        # Circle movement check
        self.recent_positions.append(tuple(self.snake_pos))
        if len(self.recent_positions) > self.max_recent:
            self.recent_positions.pop(0)

        if self.recent_positions.count(tuple(self.snake_pos)) > 2:
            reward -= 5  # Penalty for looping

        return self.get_state_as_array(), reward, done

    def get_state_as_array(self):
        grid_width = self.width // self.block_size
        grid_height = self.height // self.block_size
        state = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for pos in self.snake_body:
            gx = pos[0] // self.block_size
            gy = pos[1] // self.block_size
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                state[gy, gx, 0] = 1

        fx = self.food_pos[0] // self.block_size
        fy = self.food_pos[1] // self.block_size
        if 0 <= fx < grid_width and 0 <= fy < grid_height:
            state[fy, fx, 1] = 1

        hx = self.snake_pos[0] // self.block_size
        hy = self.snake_pos[1] // self.block_size
        if 0 <= hx < grid_width and 0 <= hy < grid_height:
            state[hy, hx, 2] = 1

        return state

    def render(self):
        self.display.fill(self.black)
        for pos in self.snake_body:
            pygame.draw.rect(self.display, self.green, pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
        pygame.draw.rect(self.display, self.white, pygame.Rect(self.food_pos[0], self.food_pos[1], self.block_size, self.block_size))
        pygame.display.update()
        self.clock.tick(self.difficulty)
