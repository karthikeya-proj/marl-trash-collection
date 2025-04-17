# environment.py
import pygame
import random

WIDTH, HEIGHT = 1200, 800
AGENT_SIZE = 40
TRASH_SIZE = 15
BIN_SIZE = 40
MOVE_SPEED = 8
ROAD_WIDTH = 80
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

class AgentSprite:
    def __init__(self, start_pos):
        self.rect = pygame.Rect(start_pos[0], start_pos[1], AGENT_SIZE, AGENT_SIZE)
        self.carrying_trash = 0
        self.max_capacity = 3

class ParkEnvironmentMultiAgent:
    def __init__(self, num_agents=2, num_trash=30, num_bins=6):
        pygame.init()
        self.num_agents = num_agents
        self.num_trash = num_trash
        self.num_bins = num_bins
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.roads = self.generate_roads()
        self.reset()

    def generate_roads(self):
        roads = []
        roads.append(pygame.Rect(0, HEIGHT//2 - ROAD_WIDTH//2, WIDTH, ROAD_WIDTH))
        roads.append(pygame.Rect(WIDTH//2 - ROAD_WIDTH//2, 0, ROAD_WIDTH, HEIGHT))
        return roads

    def generate_trash(self):
        trash = []
        while len(trash) < self.num_trash:
            x = random.randint(50, WIDTH-50)
            y = random.randint(50, HEIGHT-50)
            if not any(road.collidepoint(x, y) for road in self.roads):
                trash.append(pygame.Rect(x, y, TRASH_SIZE, TRASH_SIZE))
        return trash

    def generate_bins(self):
        bins = []
        for road in self.roads:
            for _ in range(self.num_bins//2):
                if road.width > road.height:
                    x = random.randint(road.left + 50, road.right - 50)
                    y = road.top + random.choice([-BIN_SIZE, road.height])
                else:
                    x = road.left + random.choice([-BIN_SIZE, road.width])
                    y = random.randint(road.top + 50, road.bottom - 50)
                bins.append(pygame.Rect(x, y, BIN_SIZE, BIN_SIZE))
        return bins

    def reset(self):
        self.agents = [AgentSprite((random.randint(100, WIDTH-100), random.randint(100, HEIGHT-100))) for _ in range(self.num_agents)]
        self.trash = self.generate_trash()
        self.bins = self.generate_bins()
        self.trash_collected = 0
        states = [self.get_state(agent) for agent in self.agents]
        return states

    def get_state(self, agent):
        return [agent.rect.centerx / WIDTH, agent.rect.centery / HEIGHT, agent.carrying_trash / agent.max_capacity]

    def step(self, actions):
        rewards = []
        next_states = []
        done = False

        for agent, action in zip(self.agents, actions):
            dx, dy = 0, 0
            if action == 0: dy = -MOVE_SPEED
            if action == 1: dy = MOVE_SPEED
            if action == 2: dx = -MOVE_SPEED
            if action == 3: dx = MOVE_SPEED

            agent.rect.x += dx
            agent.rect.y += dy
            agent.rect.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))

            reward = -0.1  # Small time penalty

            # Collect trash
            for t in self.trash[:]:
                if agent.rect.colliderect(t) and agent.carrying_trash < agent.max_capacity:
                    self.trash.remove(t)
                    agent.carrying_trash += 1
                    reward += 10

            # Deposit trash
            for bin_rect in self.bins:
                if agent.rect.colliderect(bin_rect) and agent.carrying_trash > 0:
                    reward += 20 * agent.carrying_trash
                    self.trash_collected += agent.carrying_trash
                    agent.carrying_trash = 0

            rewards.append(reward)
            next_states.append(self.get_state(agent))

        if self.trash_collected >= self.num_trash:
            done = True

        return next_states, rewards, done, {}

    def render(self):
        self.screen.fill(GREEN)
        for road in self.roads:
            pygame.draw.rect(self.screen, GRAY, road)
            pygame.draw.rect(self.screen, YELLOW, road.inflate(-10, -10), 2)

        for bin_rect in self.bins:
            pygame.draw.rect(self.screen, (0, 128, 0), bin_rect)

        for t in self.trash:
            pygame.draw.circle(self.screen, (255, 0, 0), t.center, TRASH_SIZE//2)

        for agent in self.agents:
            pygame.draw.rect(self.screen, (0, 0, 255), agent.rect)

        font = pygame.font.Font(None, 36)
        text = font.render(f"Collected: {self.trash_collected}/{self.num_trash}", True, WHITE)
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)
