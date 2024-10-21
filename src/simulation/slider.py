import pygame

class Slider:
    BORDER_COLOR = (250,250,250)
    SLIDER_FILL = (240,100,20)
    def __init__(self, x, y, width, height, min_value, max_value, start_val):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = self.BORDER_COLOR
        self.handle_fill = self.SLIDER_FILL
        self.handle_disabled_fill = (200, 200, 200)
        self.handle_rect = pygame.Rect(x, y, 10, height)
        self.min_val = min_value
        self._max_val = max_value
        self._value = start_val
        self.handle_pos = x + (width * (start_val - self.min_val)) / (self.max_val - self.min_val)
        self.dragging = False
        self.disabled = False

    @property
    def max_val(self):
        if self._max_val - self.min_val == 0:
            return self._max_val + 1
        return self._max_val
    
    @max_val.setter
    def max_val(self, new_max):
        self._max_val = new_max

    def draw(self, surface):
        # Draw the slider background (track)
        pygame.draw.rect(surface, self.color, self.rect, 2)

        # Draw the slider handle
        self.handle_rect.x = self.handle_pos
        color = self.handle_fill if not self.disabled else self.handle_disabled_fill
        pygame.draw.rect(surface, color, self.handle_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                # Update the position of the handle within the slider's bounds
                self.handle_pos = max(self.rect.x, min(event.pos[0], self.rect.x + self.rect.width))
                self._value = self.min_val + ((self.handle_pos - self.rect.x) / self.rect.width) * (self.max_val - self.min_val)

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, new_val):
        self._value = new_val
        self.handle_pos = self.rect.x + (self.rect.width * (new_val - self.min_val)) / (self.max_val - self.min_val)
    