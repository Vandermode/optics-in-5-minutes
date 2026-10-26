// Chromium can abort loopback assets when the host network changes. Fetch the
// actual server responses through Playwright's request API instead. maxRetries
// handles ECONNRESET only; HTTP errors, assertions and timeouts are not retried.
// Register this before media fixtures so their more-specific routes take priority.
export async function useCourseTransport(context, base) {
  await context.route(new URL(base).origin + '/**', async route => {
    try {
      await route.fulfill({response: await route.fetch({maxRetries: 2})});
    } catch (error) {
      if (!error.message.includes('closed')) throw error;
    }
  });
}
