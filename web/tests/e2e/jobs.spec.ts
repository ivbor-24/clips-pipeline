import { test, expect } from '@playwright/test';

test.describe('Jobs', () => {
  test('should show dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page.getByText('Total Jobs')).toBeVisible();
    await expect(page.getByText('Jobs')).toBeVisible();
  });

  test('should create new job', async ({ page }) => {
    await page.goto('/jobs/new');
    await expect(page.getByText('Create New Job')).toBeVisible();
    await page.getByPlaceholder('/path/to/video.mp4 or https://youtube.com/watch?v=...').fill('/dev/null');
    await page.getByRole('button', { name: 'Create Job' }).click();
    await expect(page).toHaveURL(/\/jobs\/\d+/);
  });

  test('should show job detail', async ({ page }) => {
    await page.goto('/jobs/1');
    await expect(page.getByText('Job #1')).toBeVisible();
  });
});
