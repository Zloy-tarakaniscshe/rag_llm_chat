from django.db import models


class Message(models.Model):
    user_input = models.TextField()
    ai_response = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user_input} ({self.created_at})"
