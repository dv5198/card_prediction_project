# Generated by Django 5.1.5 on 2025-01-31 16:17

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Draw',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('Club', models.CharField(max_length=2)),
                ('Diamond', models.CharField(max_length=2)),
                ('Heart', models.CharField(max_length=2)),
                ('spade', models.CharField(max_length=2)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
