import pygame

BLACK = (0, 0, 0)
RED = (122, 10, 10)
GREEN = (21, 122, 1)

class PlayPauseButton:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.is_playing = False  # Button starts in the "pause" state

    def draw(self, surface):
        # Draw the button background
        pygame.draw.rect(surface, GREEN if self.is_playing else RED, self.rect)

        # Draw the play or pause symbol
        if self.is_playing:
            # Draw the pause symbol (two vertical bars)
            bar_width = self.rect.width // 5
            spacing = self.rect.width // 6
            pygame.draw.rect(surface, BLACK, (self.rect.x + spacing, self.rect.y + 10, bar_width, self.rect.height - 20))
            pygame.draw.rect(surface, BLACK, (self.rect.x + 3 * spacing, self.rect.y + 10, bar_width, self.rect.height - 20))
        else:
            # Draw the play symbol (triangle)
            pygame.draw.polygon(surface, BLACK, [
                (self.rect.x + 10, self.rect.y + 10),
                (self.rect.x + self.rect.width - 10, self.rect.y + self.rect.height // 2),
                (self.rect.x + 10, self.rect.y + self.rect.height - 10)
            ])

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_playing = not self.is_playing  # Toggle the state on click

    def get_state(self):
        return self.is_playing