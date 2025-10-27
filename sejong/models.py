from django.db import models

# Create your models here.
class ImageNas(models.Model):
    image_nas_id = models.AutoField(primary_key=True) # 기본키
    user_id = models.IntegerField()
    experiment_key = models.CharField(max_length=100)
    dataset = models.CharField(max_length=100)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    search_algorithm = models.CharField(max_length=100)
    layer_candidate = models.CharField(max_length=100)
    max_epochs = models.IntegerField()
    batch_size = models.IntegerField()
    learning_rate = models.FloatField()
    momentum = models.FloatField()
    weight_decay = models.FloatField()
    auxiliary_loss_weight = models.FloatField()
    width = models.IntegerField()
    num_cells = models.IntegerField()
    drop_path_prob = models.IntegerField()
    use_yn = models.CharField(max_length=1)

class MultimodalNas(models.Model):
    multimodal_nas_id = models.AutoField(primary_key=True) # 기본키
    user_id = models.IntegerField()
    experiment_key = models.CharField(max_length=100)
    dataset = models.CharField(max_length=100)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    performance = models.CharField(max_length=100)
    max_epochs = models.IntegerField()
    batch_size = models.IntegerField()
    learning_rate = models.FloatField()
    optimizer = models.CharField(max_length=100)
    lr_scheduler = models.CharField(max_length=100)
    weight_decay = models.FloatField()
    warmup_epochs = models.FloatField()
    label_smoothing = models.FloatField()
    drop_path = models.FloatField()
    use_yn = models.CharField(max_length=1)
    
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

