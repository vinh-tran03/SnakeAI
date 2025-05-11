import pygame, sys, random
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

        self.reset()

    def reset(self):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = [random.randrange(1, self.width // self.block_size) * self.block_size,
                         random.randrange(1, self.height // self.block_size) * self.block_size]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        return self.get_state_as_array()

    def step(self, action):
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        new_direction = directions[action]

        # Prevent direct reversal of direction
        if (new_direction == 'UP' and self.direction != 'DOWN') or \
           (new_direction == 'DOWN' and self.direction != 'UP') or \
           (new_direction == 'LEFT' and self.direction != 'RIGHT') or \
           (new_direction == 'RIGHT' and self.direction != 'LEFT'):
            self.direction = new_direction

        # Move snake
        if self.direction == 'UP':
            self.snake_pos[1] -= self.block_size
        elif self.direction == 'DOWN':
            self.snake_pos[1] += self.block_size
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= self.block_size
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += self.block_size

        self.snake_body.insert(0, list(self.snake_pos))

        reward = -0.01  # Small penalty to discourage inaction
        done = False

        # Eating food
        if self.snake_pos == self.food_pos:
            self.score += 1
            reward = 100  # Reward for eating food
            self.food_spawn = False
        else:
            self.snake_body.pop()

        # Respawn food
        if not self.food_spawn:
            self.food_pos = [random.randrange(1, self.width // self.block_size) * self.block_size,
                             random.randrange(1, self.height // self.block_size) * self.block_size]
        self.food_spawn = True

        # Check for collision (with wall or self)
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.width or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= self.height or
            self.snake_pos in self.snake_body[1:]):
            done = True
            reward = -10  # Heavy penalty for dying

        return self.get_state_as_array(), reward, done

    def get_state_as_array(self):
        grid_width = self.width // self.block_size
        grid_height = self.height // self.block_size
        state = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Snake body
        for pos in self.snake_body:
            gx = pos[0] // self.block_size
            gy = pos[1] // self.block_size
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                state[gy, gx, 0] = 1

        # Food
        fx = self.food_pos[0] // self.block_size
        fy = self.food_pos[1] // self.block_size
        if 0 <= fx < grid_width and 0 <= fy < grid_height:
            state[fy, fx, 1] = 1

        # Snake head
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
