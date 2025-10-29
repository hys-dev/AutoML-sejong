from django.db import models

# Create your models here.

class UploadedZip(models.Model):
    CATEGORY_CHOICES = [
        ('image', 'Image NAS'),
        ('multi', 'Multi-modal NAS'),
    ]
    category = models.CharField(max_length=10, choices=CATEGORY_CHOICES)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.category}: {self.file.name}"