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

class ImageNas(models.Model):
    image_nas_id = models.AutoField(primary_key=True)
    user_id = models.IntegerField(null=True) # auth_user 테이블의 id(cd)
    exp_key = models.CharField() # 각 실험을 특정하는 랜덤 문자열
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True)
    dataset_name = models.CharField(max_length=100)
    layer_candidates = models.JSONField()
    max_epochs = models.IntegerField()
    strategy = models.CharField(max_length=100)
    batch_size = models.IntegerField()
    learning_rate = models.FloatField()
    momentum = models.FloatField()
    weight_decay = models.FloatField()
    gradient_clip_val = models.FloatField()
    width = models.IntegerField()
    num_of_cells = models.IntegerField()
    aux_loss_weight = models.FloatField()
    drop_path_prob = models.CharField(null=True)
    use_yn = models.CharField(max_length=1, default="y")

class MultimodalNas(models.Model):
    multimodal_nas_id = models.AutoField(primary_key=True)
    user_id = models.IntegerField()  # auth_user 테이블의 id
    exp_key = models.CharField()  # 각 실험을 특정하는 랜덤 문자열
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField()
    max_epochs = models.IntegerField()
    batch_size = models.IntegerField()
    learning_rate = models.FloatField()
    min_learning_rate = models.FloatField()
    warmup_epochs = models.IntegerField()
    weight_decay = models.FloatField()
    optimizer = models.CharField(max_length=100)
    lr_scheduler = models.CharField(max_length=100)
    use_yn = models.CharField(max_length=1)

class MultimodalEvo(models.Model):
    evo_id = models.AutoField(primary_key=True)
    exp_key = models.CharField()
    multimodal_nas_id = models.CharField()
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField()
    max_epochs = models.IntegerField()
    batch_size = models.IntegerField()
    min_param_limits = models.IntegerField()
    param_limits = models.IntegerField()
    select_num = models.IntegerField()
    crossover_num = models.IntegerField()
    mutation_num = models.IntegerField()
