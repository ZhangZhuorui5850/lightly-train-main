SELECT *
FROM (
    VALUES
        ('train', 'small', 81.4, 156219, 12883),
        ('train', 'medium', 15.4, 156219, 12883),
        ('train', 'large', 3.2, 156219, 12883),
        ('val', 'small', 79.0, 16995, 1596),
        ('val', 'medium', 17.3, 16995, 1596),
        ('val', 'large', 3.7, 16995, 1596),
        ('test', 'small', 84.0, 22866, 1615),
        ('test', 'medium', 13.4, 22866, 1615),
        ('test', 'large', 2.6, 22866, 1615)
) AS profile(split, size_class, box_share_pct, total_boxes, total_images);
